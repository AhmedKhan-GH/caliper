#include <caliper_builder/session/spec_edit.h>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/patch.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <set>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;
using spec::LayoutNode;

// Layout trees of a valid spec are at most 8 nodes deep; this bounds the
// recursion of the helpers below for any tree.
constexpr int kMaxWalkDepth = 300;

void erase_x(json& object) {
    if (object.is_object()) object.erase("x");
}

std::string index_path(const char* noun, std::size_t i) { return std::string(noun) + "[" + std::to_string(i) + "]"; }

Error fail(std::string code, std::string message, std::string where = {}) {
    return Error(std::move(code), std::move(message), std::move(where));
}

std::string binding_source(const std::string& binding) {
    const auto dot = binding.find('.');
    return dot == std::string::npos ? binding : binding.substr(0, dot);
}

// Rounds a share to 12 decimals, so that 1 - 0.4 is written as 0.6.
double tidy_ratio(double r) { return std::round(r * 1e12) / 1e12; }

bool layout_places(const LayoutNode& node, const std::string& panel, int depth = 0) {
    if (depth > kMaxWalkDepth) return false;
    if (node.kind == LayoutNode::Kind::panel) return node.panel == panel;
    for (const auto& c : node.children)
        if (layout_places(c, panel, depth + 1)) return true;
    return false;
}

// Removes the leaf of `panel`. A split left with one child is replaced by
// that child, and a split or tabs left with none is removed itself.
bool remove_leaf(LayoutNode& node, const std::string& panel, bool* remove_self, int depth = 0) {
    *remove_self = false;
    if (depth > kMaxWalkDepth) return false;
    if (node.kind == LayoutNode::Kind::panel) {
        if (node.panel != panel) return false;
        *remove_self = true;
        return true;
    }
    for (std::size_t i = 0; i < node.children.size(); ++i) {
        bool drop = false;
        if (!remove_leaf(node.children[i], panel, &drop, depth + 1)) continue;
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

// Puts a new leaf for `panel` next to the leaf of `target`.
bool place_next_to(LayoutNode& node, const std::string& target, const std::string& panel, const std::string& where,
                   double share, int depth = 0) {
    if (depth > kMaxWalkDepth) return false;
    if (node.kind == LayoutNode::Kind::panel) {
        if (node.panel != target) return false;
        LayoutNode old = std::move(node);
        LayoutNode wrap;
        if (where == "tabbed") {
            wrap.kind = LayoutNode::Kind::tabs;
            wrap.children.push_back(std::move(old));
            wrap.children.push_back(LayoutNode::of_panel(panel));
        } else {
            wrap.kind = LayoutNode::Kind::split;
            wrap.split = where == "beside" ? spec::SplitDirection::horizontal : spec::SplitDirection::vertical;
            const bool new_first = where == "above";
            if (new_first) {
                wrap.children.push_back(LayoutNode::of_panel(panel));
                wrap.children.push_back(std::move(old));
            } else {
                wrap.children.push_back(std::move(old));
                wrap.children.push_back(LayoutNode::of_panel(panel));
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
                                     LayoutNode::of_panel(panel));
                return true;
            }
        }
    }
    for (auto& c : node.children)
        if (place_next_to(c, target, panel, where, share, depth + 1)) return true;
    return false;
}

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

json without_dry_run(const json& args) {
    json out = args.is_object() ? args : json::object();
    out.erase("dry_run");
    return out;
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

SpecEdit source_add(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    if (s.find_source(id))
        return refuse(fail("source.exists", "The spec already has a source " + id + "; choose another id.", "args.id"));
    auto decoded = spec::source_from_json(args, index_path("source", s.sources.size()));
    if (!decoded.ok()) return refuse(first_spec_error(decoded.diagnostics));
    spec::Spec next = s;
    next.sources.push_back(std::move(decoded.value));
    return finish(std::move(next), resolver, json{{"id", id}});
}

// Maps the patch library's refusals to the command layer's codes.
Error patch_refusal(const std::vector<spec::Diagnostic>& diagnostics, const char* noun) {
    Error err = first_spec_error(diagnostics);
    const std::string n(noun);
    if (err.code == "spec.not_found") return fail(n + ".not_found", err.message, "args.id");
    if (err.code == "spec.bad_patch" && err.where.size() > 3 && err.where.compare(err.where.size() - 3, 3, ".id") == 0)
        return fail(n + ".id_immutable", err.message, "args.patch.id");
    return err;
}

SpecEdit source_patch(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const json patch = args.contains("patch") ? args["patch"] : json::object();
    spec::PatchResult r = spec::patch_source(s, id, patch, resolver);
    if (!r.applied || !r.ok()) return refuse(patch_refusal(r.diagnostics, "source"));
    const spec::Source* src = r.spec.find_source(id);
    json result{{"source", src ? source_result_json(*src) : json::object()}};
    SpecEdit out;
    for (auto& d : r.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    out.spec = std::move(r.spec);
    out.result = std::move(result);
    return out;
}

SpecEdit source_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    std::size_t index = s.sources.size();
    for (std::size_t i = 0; i < s.sources.size(); ++i)
        if (s.sources[i].id == id) index = i;
    if (index == s.sources.size()) return refuse(fail("source.not_found", "The spec has no source " + id + ".", "args.id"));
    for (std::size_t p = 0; p < s.panels.size(); ++p)
        for (const auto& [slot, value] : s.panels[p].bind)
            for (const auto& b : value.bindings)
                if (binding_source(b) == id)
                    return refuse(fail("source.in_use",
                                       "The panel " + s.panels[p].id + " binds the source " + id +
                                           "; change or remove that panel first.",
                                       index_path("panel", p) + ".bind." + slot));
    for (std::size_t i = 0; i < s.sources.size(); ++i)
        if (s.sources[i].from && *s.sources[i].from == id)
            return refuse(fail("source.in_use",
                               "The source " + s.sources[i].id + " derives from " + id + "; change or remove it first.",
                               index_path("source", i) + ".from"));
    spec::Spec next = s;
    next.sources.erase(next.sources.begin() + static_cast<std::ptrdiff_t>(index));
    return finish(std::move(next), resolver, json{{"id", id}});
}

// Whether every leaf under `node` names a control of the spec.
bool only_controls(const LayoutNode& node, const spec::Spec& s, int depth = 0) {
    if (depth > kMaxWalkDepth) return false;
    if (node.kind == LayoutNode::Kind::panel) return s.find_control(node.panel) != nullptr;
    if (node.children.empty()) return false;
    for (const auto& c : node.children)
        if (!only_controls(c, s, depth + 1)) return false;
    return true;
}

SpecEdit panel_create(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    if (s.find_panel(id))
        return refuse(fail("panel.exists", "The spec already has a panel " + id + "; choose another id.", "args.id"));
    json item = args;
    item.erase("place");
    auto decoded = spec::panel_from_json(item, index_path("panel", s.panels.size()));
    if (!decoded.ok()) return refuse(first_spec_error(decoded.diagnostics));

    std::optional<LayoutNode> layout = effective_layout(s);
    if (args.contains("place") && args["place"].is_object()) {
        const json& place = args["place"];
        const std::string target = str_arg(place, "target");
        const std::string where = str_arg(place, "where");
        const double share = place.contains("ratio") && place["ratio"].is_number() ? place["ratio"].get<double>() : 0.5;
        if (!layout || !layout_places(*layout, target))
            return refuse(fail("panel.not_found", "The layout has no panel " + target + " to place the new panel next to.",
                               "args.place.target"));
        place_next_to(*layout, target, id, where, share);
    } else if (!layout) {
        layout = LayoutNode::of_panel(id);
    } else {
        // Beside the panels; a row of controls at the bottom of the layout stays under them.
        LayoutNode* panels = &*layout;
        if (layout->kind == LayoutNode::Kind::split && layout->split == spec::SplitDirection::vertical &&
            layout->children.size() == 2 && only_controls(layout->children[1], s)) {
            if (only_controls(layout->children[0], s)) {
                LayoutNode wrap;
                wrap.kind = LayoutNode::Kind::split;
                wrap.split = spec::SplitDirection::vertical;
                wrap.children.push_back(LayoutNode::of_panel(id));
                wrap.children.push_back(std::move(*layout));
                layout = std::move(wrap);
                panels = nullptr;
            } else {
                panels = &layout->children[0];
            }
        } else if (only_controls(*layout, s)) {
            LayoutNode wrap;
            wrap.kind = LayoutNode::Kind::split;
            wrap.split = spec::SplitDirection::vertical;
            wrap.children.push_back(LayoutNode::of_panel(id));
            wrap.children.push_back(std::move(*layout));
            layout = std::move(wrap);
            panels = nullptr;
        }
        if (panels) {
            LayoutNode wrap;
            wrap.kind = LayoutNode::Kind::split;
            wrap.split = spec::SplitDirection::horizontal;
            wrap.children.push_back(std::move(*panels));
            wrap.children.push_back(LayoutNode::of_panel(id));
            *panels = std::move(wrap);
        }
    }
    spec::Spec next = s;
    next.panels.push_back(std::move(decoded.value));
    next.layout = std::move(layout);
    // A scene panel raises a version 0 spec to version 1 in the same step.
    next.version = std::max(next.version, spec::required_version(next));
    return finish(std::move(next), resolver, json{{"id", id}});
}

SpecEdit panel_patch(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    const json patch = args.contains("patch") ? args["patch"] : json::object();
    spec::PatchResult r = spec::patch_panel(s, id, patch, resolver);
    if (!r.applied || !r.ok()) return refuse(patch_refusal(r.diagnostics, "panel"));
    const spec::Panel* panel = r.spec.find_panel(id);
    json result{{"panel", panel ? panel_result_json(*panel) : json::object()}};
    SpecEdit out;
    for (auto& d : r.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    out.spec = std::move(r.spec);
    out.result = std::move(result);
    return out;
}

SpecEdit panel_remove(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string id = str_arg(args, "id");
    std::size_t index = s.panels.size();
    for (std::size_t i = 0; i < s.panels.size(); ++i)
        if (s.panels[i].id == id) index = i;
    if (index == s.panels.size()) return refuse(fail("panel.not_found", "The spec has no panel " + id + ".", "args.id"));
    spec::Spec next = s;
    next.panels.erase(next.panels.begin() + static_cast<std::ptrdiff_t>(index));
    if (next.layout) {
        bool drop = false;
        remove_leaf(*next.layout, id, &drop);
        if (drop) next.layout.reset();
    }
    return finish(std::move(next), resolver, json{{"id", id}});
}

SpecEdit layout_place(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    const std::string panel = str_arg(args, "panel");
    const std::string target = str_arg(args, "target");
    const std::string where = str_arg(args, "where");
    const double share = args.contains("ratio") && args["ratio"].is_number() ? args["ratio"].get<double>() : 0.5;
    // From spec version 1 a leaf names a panel or a control.
    if (!s.find_panel(panel) && !s.find_control(panel))
        return refuse(fail("panel.not_found", "The spec has no panel or control " + panel + ".", "args.panel"));
    if (panel == target)
        return refuse(fail("layout.same_panel", "A panel cannot be placed next to itself.", "args.target"));
    if (!s.find_panel(target) && !s.find_control(target))
        return refuse(fail("panel.not_found", "The spec has no panel or control " + target + ".", "args.target"));
    std::optional<LayoutNode> layout = effective_layout(s);
    if (!layout || !layout_places(*layout, target))
        return refuse(fail("panel.not_found", "The panel " + target + " is not in the layout, so nothing can be placed next to it.",
                           "args.target"));
    bool drop = false;
    remove_leaf(*layout, panel, &drop);
    // The target is another panel, so the layout keeps at least its leaf.
    if (drop) return refuse(fail("layout.invalid", "The layout would be empty.", "args.panel"));
    place_next_to(*layout, target, panel, where, share);
    spec::Spec next = s;
    next.layout = std::move(layout);
    json flat = layout_to_flat(*next.layout, next);
    return finish(std::move(next), resolver, json{{"layout", std::move(flat)}});
}

SpecEdit layout_set(const spec::Spec& s, const json& args, const spec::SourceResolver& resolver) {
    FlatLayoutParse parsed = layout_from_flat(args.contains("layout") ? args["layout"] : json(), s);
    if (parsed.error) return refuse(std::move(parsed.error));
    spec::Spec next = s;
    next.layout = std::move(parsed.root);
    json flat = layout_to_flat(*next.layout, next);
    return finish(std::move(next), resolver, json{{"layout", std::move(flat)}});
}

SpecEdit set_text(const json& args, const spec::SourceResolver& resolver) {
    SpecTextCheck checked = check_spec_text(str_arg(args, "text"), resolver);
    if (checked.error) return refuse(std::move(checked.error));
    SpecEdit out;
    for (auto& d : checked.diagnostics)
        if (!d.is_error()) out.warnings.push_back(std::move(d));
    out.result = json{{"id", checked.spec.applet.id}};
    out.spec = std::move(checked.spec);
    return out;
}

// Flattening: assigns node ids in pre-order.
struct Flattener {
    const std::set<std::string>& panels;
    json nodes = json::array();
    int next = 1;

    std::string fresh_id() {
        for (;;) {
            std::string id = "n" + std::to_string(next++);
            if (!panels.count(id)) return id;
        }
    }

    std::string visit(const LayoutNode& node, int depth) {
        if (node.kind == LayoutNode::Kind::panel || depth > kMaxWalkDepth) return node.panel;
        const std::string id = fresh_id();
        const std::size_t slot = nodes.size();
        nodes.push_back(json::object());
        json children = json::array();
        for (const auto& c : node.children) children.push_back(visit(c, depth + 1));
        json n{{"id", id},
               {"split", node.kind == LayoutNode::Kind::tabs ? "tabs" : spec::to_string(node.split)},
               {"ratio", node.kind == LayoutNode::Kind::split && node.ratio ? json(*node.ratio) : json(nullptr)},
               {"children", std::move(children)}};
        if (node.title) n["title"] = *node.title;
        nodes[slot] = std::move(n);
        return id;
    }
};

struct Unflattener {
    const std::set<std::string>& panels;
    std::map<std::string, std::size_t> index;  // node id to its position in `nodes`
    const json& nodes;
    std::vector<bool> used;
    Error error;

    bool build(const std::string& name, const std::string& where, int depth, LayoutNode& out) {
        const auto it = index.find(name);
        if (it == index.end()) {
            if (!panels.count(name)) {
                error = fail("panel.not_found", "The layout names " + name + ", which is neither a node of the layout nor a panel of the spec.",
                             where);
                return false;
            }
            out = LayoutNode::of_panel(name);
            return true;
        }
        if (used[it->second] || depth > kMaxWalkDepth) {
            error = fail("layout.invalid", "The node " + name + " is used more than once, or the nodes form a cycle.", where);
            return false;
        }
        used[it->second] = true;
        const json& node = nodes[it->second];
        const std::string base = "args.layout.nodes[" + std::to_string(it->second) + "]";
        const std::string split = node.value("split", std::string());
        const bool has_ratio = node.contains("ratio") && node["ratio"].is_number();
        if (split == "tabs") {
            if (has_ratio) {
                error = fail("layout.invalid", "Tabs take no ratio; set it to null.", base + ".ratio");
                return false;
            }
            out.kind = LayoutNode::Kind::tabs;
        } else {
            out.kind = LayoutNode::Kind::split;
            if (!spec::split_from_string(split, out.split)) {
                error = fail("layout.invalid", "A node splits horizontal, vertical or tabs.", base + ".split");
                return false;
            }
            if (has_ratio) out.ratio = node["ratio"].get<double>();
        }
        if (node.contains("title") && node["title"].is_string()) out.title = node["title"].get<std::string>();
        const json& children = node.contains("children") ? node["children"] : json::array();
        for (std::size_t i = 0; i < children.size(); ++i) {
            if (!children[i].is_string()) {
                error = fail("layout.invalid", "A child is a panel id or a node id.", base + ".children[" + std::to_string(i) + "]");
                return false;
            }
            LayoutNode child;
            if (!build(children[i].get<std::string>(), base + ".children[" + std::to_string(i) + "]", depth + 1, child))
                return false;
            out.children.push_back(std::move(child));
        }
        return true;
    }
};

// The ids a layout leaf may name: the panels and, from spec version 1, the
// controls.
std::set<std::string> panel_ids(const spec::Spec& s) {
    std::set<std::string> out;
    for (const auto& p : s.panels) out.insert(p.id);
    for (const auto& c : s.controls) out.insert(c.id);
    return out;
}

}  // namespace

json source_command_args(const spec::Source& source) {
    json out = spec::to_json(source);
    erase_x(out);
    if (out.contains("transform"))
        for (auto& t : out["transform"]) erase_x(t);
    return out;
}

json panel_command_args(const spec::Panel& panel) {
    json out = spec::to_json(panel);
    erase_x(out);
    if (out.contains("options")) {
        erase_x(out["options"]);
        if (out["options"].empty()) out.erase("options");
    }
    return out;
}

json panel_result_json(const spec::Panel& panel) {
    json out = panel_command_args(panel);
    if (!out.contains("title")) out["title"] = nullptr;
    if (!out.contains("bind")) out["bind"] = json::object();
    if (!out.contains("options")) out["options"] = json::object();
    return out;
}

json source_result_json(const spec::Source& source) { return source_command_args(source); }

std::optional<LayoutNode> effective_layout(const spec::Spec& s) {
    if (s.layout) return s.layout;
    // The panels as version 0 places them: one alone, or a vertical split.
    std::optional<LayoutNode> panels;
    if (s.panels.size() == 1) {
        panels = LayoutNode::of_panel(s.panels[0].id);
    } else if (!s.panels.empty()) {
        LayoutNode split;
        split.kind = LayoutNode::Kind::split;
        split.split = spec::SplitDirection::vertical;
        for (const auto& p : s.panels) split.children.push_back(LayoutNode::of_panel(p.id));
        panels = std::move(split);
    }
    if (s.controls.empty()) return panels;
    // Version 1: one row of the controls in declaration order, under the panels.
    std::optional<LayoutNode> row;
    if (s.controls.size() == 1) {
        row = LayoutNode::of_panel(s.controls[0].id);
    } else {
        LayoutNode split;
        split.kind = LayoutNode::Kind::split;
        split.split = spec::SplitDirection::horizontal;
        for (const auto& c : s.controls) split.children.push_back(LayoutNode::of_panel(c.id));
        row = std::move(split);
    }
    if (!panels) return row;
    LayoutNode root;
    root.kind = LayoutNode::Kind::split;
    root.split = spec::SplitDirection::vertical;
    root.children.push_back(std::move(*panels));
    root.children.push_back(std::move(*row));
    return root;
}

json layout_to_flat(const LayoutNode& root, const spec::Spec& s) {
    try {
        const std::set<std::string> panels = panel_ids(s);
        Flattener f{panels};
        const std::string top = f.visit(root, 0);
        return json{{"root", top}, {"nodes", std::move(f.nodes)}};
    } catch (const std::exception&) {
        return json{{"root", ""}, {"nodes", json::array()}};
    }
}

FlatLayoutParse layout_from_flat(const json& flat, const spec::Spec& s) {
    FlatLayoutParse out;
    try {
        if (!flat.is_object() || !flat.contains("root") || !flat["root"].is_string() || !flat.contains("nodes") ||
            !flat["nodes"].is_array()) {
            out.error = fail("layout.invalid", "A layout is {\"root\": \"<id>\", \"nodes\": [...]}.", "args.layout");
            return out;
        }
        const std::set<std::string> panels = panel_ids(s);
        const json& nodes = flat["nodes"];
        Unflattener u{panels, {}, nodes, std::vector<bool>(nodes.size(), false), {}};
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const std::string where = "args.layout.nodes[" + std::to_string(i) + "].id";
            if (!nodes[i].is_object() || !nodes[i].contains("id") || !nodes[i]["id"].is_string()) {
                out.error = fail("layout.invalid", "Each node has an id.", where);
                return out;
            }
            const std::string id = nodes[i]["id"].get<std::string>();
            if (panels.count(id)) {
                out.error = fail("layout.invalid", "The node id " + id + " is also a panel's id; node ids must differ from panel ids.", where);
                return out;
            }
            if (!u.index.emplace(id, i).second) {
                out.error = fail("layout.invalid", "Two nodes have the id " + id + ".", where);
                return out;
            }
        }
        if (!u.build(flat["root"].get<std::string>(), "args.layout.root", 0, out.root)) {
            out.error = std::move(u.error);
            return out;
        }
        for (std::size_t i = 0; i < nodes.size(); ++i)
            if (!u.used[i]) {
                out.error = fail("layout.invalid", "The node " + nodes[i]["id"].get<std::string>() + " cannot be reached from the root.",
                                 "args.layout.nodes[" + std::to_string(i) + "]");
                return out;
            }
    } catch (const std::exception& e) {
        out.error = fail("layout.invalid", std::string("The layout could not be read: ") + e.what(), "args.layout");
    }
    return out;
}

json diagnostic_json(const spec::Diagnostic& d) {
    const Error e = d.to_error();
    return json{{"code", e.code}, {"message", e.message}, {"where", e.where}};
}

json warnings_json(const std::vector<spec::Diagnostic>& diagnostics) {
    json out = json::array();
    for (const auto& d : diagnostics)
        if (!d.is_error()) out.push_back(diagnostic_json(d));
    return out;
}

Error first_spec_error(const std::vector<spec::Diagnostic>& diagnostics) {
    for (const auto& d : diagnostics)
        if (d.is_error()) return d.to_error();
    return {};
}

bool is_spec_edit_command(std::string_view name) {
    return name == "spec.set_text" || name == "source.add" || name == "source.patch" || name == "source.remove" ||
           name == "panel.create" || name == "panel.patch" || name == "panel.remove" || name == "layout.place" ||
           name == "layout.set" || name == "control.create" || name == "control.patch" || name == "control.remove";
}

SpecEdit apply_spec_edit(const spec::Spec& s, std::string_view name, const json& raw_args,
                         const spec::SourceResolver& resolver) {
    try {
        const json args = without_dry_run(raw_args);
        if (name == "spec.set_text") return set_text(args, resolver);
        if (name == "source.add") return source_add(s, args, resolver);
        if (name == "source.patch") return source_patch(s, args, resolver);
        if (name == "source.remove") return source_remove(s, args, resolver);
        if (name == "panel.create") return panel_create(s, args, resolver);
        if (name == "panel.patch") return panel_patch(s, args, resolver);
        if (name == "panel.remove") return panel_remove(s, args, resolver);
        if (name == "layout.place") return layout_place(s, args, resolver);
        if (name == "layout.set") return layout_set(s, args, resolver);
        if (name == "control.create" || name == "control.patch" || name == "control.remove")
            return apply_control_edit(s, name, args, resolver);
        return refuse(fail("command.unknown", "This is not a command that edits the open spec.", std::string(name)));
    } catch (const std::exception& e) {
        return refuse(fail("spec.internal", std::string("The edit failed unexpectedly: ") + e.what()));
    }
}

spec::Spec new_spec_from_args(const json& args) {
    spec::Spec s;
    s.applet.id = args.value("id", std::string());
    if (args.contains("title") && args["title"].is_string()) s.applet.title = args["title"].get<std::string>();
    if (args.contains("description") && args["description"].is_string())
        s.applet.description = args["description"].get<std::string>();
    return s;
}

SpecTextCheck check_spec_text(std::string_view text, const spec::SourceResolver& resolver) {
    SpecTextCheck out;
    try {
        spec::ParseResult parsed = spec::parse_toml(text);
        if (!parsed.ok()) {
            out.diagnostics = std::move(parsed.diagnostics);
            out.error = first_spec_error(out.diagnostics);
            return out;
        }
        spec::ValidationReport report = spec::validate(parsed.spec, resolver);
        spec::locate(report.diagnostics, parsed.positions);
        out.diagnostics = std::move(report.diagnostics);
        out.error = first_spec_error(out.diagnostics);
        if (!out.error) out.spec = std::move(parsed.spec);
    } catch (const std::exception& e) {
        out.error = fail("spec.internal", std::string("The spec could not be checked: ") + e.what());
    }
    return out;
}

}  // namespace caliper_builder
