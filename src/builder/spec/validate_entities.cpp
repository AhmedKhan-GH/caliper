// The validator's checks of the entities of a scene panel and of the scene
// options `box` and `views` (design/instrument-interface.md, Scene Entities,
// Data Box, and Starting Camera and Saved Views).
//
// The codec has read the keys of each entity, its draw kind, the form of its
// `bind` and its pose. This check adds what needs the rest of the spec: the
// path grammar and its uniqueness in the panel, the sources and slots that
// `bind` names, the options of each draw kind with their types and ranges,
// and the limits. A reference at a value place is left to
// validate_references.cpp, which checks its target and type.
#include "validate_additions.h"

#include <caliper_builder/spec/colormaps.h>
#include <caliper_builder/spec/entity.h>

#include <cmath>
#include <set>

namespace caliper_builder::spec::detail {

namespace {

// The slots of each draw kind, in the order of the plan's table.
const std::vector<std::string>& entity_slots(DrawKind k) {
    static const std::vector<std::string> points = {"positions", "x", "y", "z", "values", "colors", "radii"};
    static const std::vector<std::string> lines = {"positions", "x", "y", "z", "values", "colors"};
    static const std::vector<std::string> surface = {"height", "values", "x", "y"};
    switch (k) {
        case DrawKind::points: return points;
        case DrawKind::lines: return lines;
        case DrawKind::surface: return surface;
    }
    return points;
}

// The options of each draw kind.
const std::vector<std::string>& entity_options(DrawKind k) {
    static const std::vector<std::string> points = {"colormap", "vmin",  "vmax", "size_px",      "shape",     "x",
                                                    "y",        "z",     "own",  "depth_offset", "highlight", "visible"};
    static const std::vector<std::string> lines = {"colormap", "vmin", "vmax",         "mode",      "width_px", "x", "y",
                                                   "z",        "own",  "depth_offset", "highlight", "visible"};
    static const std::vector<std::string> surface = {"colormap",     "vmin",      "vmax",
                                                     "depth_offset", "highlight", "visible"};
    switch (k) {
        case DrawKind::points: return points;
        case DrawKind::lines: return lines;
        case DrawKind::surface: return surface;
    }
    return points;
}

bool path_segment_char(char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-'; }

// 1 to 8 segments of [a-z0-9_-], joined by "/".
bool valid_entity_path(const std::string& p) {
    if (p.empty() || p.size() > 255) return false;
    std::size_t segments = 1, run = 0;
    for (char c : p) {
        if (c == '/') {
            if (run == 0) return false;
            ++segments;
            run = 0;
        } else if (path_segment_char(c)) {
            ++run;
        } else {
            return false;
        }
    }
    return run > 0 && segments <= limits::kEntityPathSegments;
}

bool has_source(const Spec& s, const std::string& id) {
    for (const Source& src : s.sources)
        if (src.id == id) return true;
    return false;
}

class EntityCheck {
public:
    explicit EntityCheck(const AdditionsCheck& c) : c_(c) {}

    void run() {
        for (std::size_t i = 0; i < c_.spec.panels.size(); ++i) {
            const Panel& p = c_.spec.panels[i];
            const std::string path = index_path("panel", i);
            entities(p, path);
            box(p, path);
            views(p, path);
        }
    }

private:
    void error(const char* code, const std::string& message, const std::string& path) {
        c_.sink.error(code, message, path);
    }

    void entities(const Panel& p, const std::string& path) {
        if (p.entities.empty()) return;
        const std::string ep = path + ".entity";
        if (p.primitive != Primitive::scene) {
            error("spec.bad_value", "Only a scene panel draws entities; " + quote(p.id) + " is not a scene panel.",
                  ep);
            return;
        }
        if (p.entities.size() > limits::kEntities) {
            error("spec.too_many",
                  "A scene panel draws at most " + std::to_string(limits::kEntities) + " entities; " + quote(p.id) +
                      " has " + std::to_string(p.entities.size()) + ".",
                  ep);
            return;
        }
        std::set<std::string> seen;
        for (std::size_t j = 0; j < p.entities.size(); ++j) {
            const Entity& e = p.entities[j];
            const std::string at = index_path(ep, j);
            if (!valid_entity_path(e.path)) {
                error("spec.bad_value",
                      quote(e.path) +
                          " is not an entity path; a path is 1 to 8 segments of a-z, 0-9, _ and -, joined by /.",
                      at + ".path");
            } else if (!seen.insert(e.path).second) {
                error("spec.duplicate_id", "Two entities of the panel " + quote(p.id) + " have the path " +
                                               quote(e.path) + ".",
                      at + ".path");
            }
            if (e.title && (e.title->empty() || code_points(*e.title) > 128 || has_control(*e.title, TextKind::line)))
                error("spec.bad_value", "An entity's title is 1 to 128 characters on one line.", at + ".title");
            bind(e, at);
            options(e, at);
            pose(e, at);
        }
    }

    void binding(const std::string& b, const std::string& at) {
        const std::size_t dot = b.find('.');
        const std::string src = b.substr(0, dot);
        if (!valid_id(src) || (dot != std::string::npos && dot + 1 >= b.size())) {
            error("spec.bad_binding", quote(b) + " is not a binding; a binding is <source> or <source>.<column>.",
                  at);
            return;
        }
        if (!has_source(c_.spec, src))
            error("spec.unknown_source",
                  "The binding " + quote(b) + " names " + quote(src) + ", which is not a source of this spec.", at);
    }

    void bind(const Entity& e, const std::string& at) {
        const std::string bp = at + ".bind";
        if (e.source) {
            binding(*e.source, bp);
            return;
        }
        if (e.slots.empty()) {
            error("spec.missing_slot", "An entity binds a source id or a table of slots.", bp);
            return;
        }
        const std::vector<std::string>& known = entity_slots(e.draw);
        for (const auto& [slot, value] : e.slots) {
            const std::string sp = key_path(bp, slot);
            if (!contains(known, slot)) {
                error("spec.unknown_slot",
                      "A " + std::string(draw_kind_name(e.draw)) + " entity has no slot " + quote(slot) +
                          "; its slots are " + join(known) + ".",
                      sp);
                continue;
            }
            if (const std::string* b = std::get_if<std::string>(&value)) binding(*b, sp);
        }
        auto has = [&](const char* s) { return e.slots.count(s) > 0; };
        if (e.draw == DrawKind::surface) {
            if (!has("height")) error("spec.missing_slot", "A surface entity needs the slot height.", bp);
        } else if (!has("positions") && int(has("x")) + int(has("y")) + int(has("z")) < 2) {
            error("spec.missing_slot", "A " + std::string(draw_kind_name(e.draw)) +
                                           " entity needs the slot positions, or at least two of x, y and z.",
                  bp);
        }
        if (has("colors") && has("values"))
            error("spec.slot_mismatch", "An entity takes its colour from colors or from values, not both.", bp);
        if (has("positions") && (has("x") || has("y") || has("z")))
            error("spec.slot_mismatch", "An entity takes positions or x, y and z, not both.", bp);
    }

    void number_range(const nlohmann::json& v, const std::string& at, double lo, double hi, const char* what) {
        if (!v.is_number()) {
            error("spec.bad_value", std::string(what) + " is a number.", at);
        } else if (const double d = v.get<double>(); !(d >= lo && d <= hi)) {
            error("spec.out_of_range",
                  std::string(what) + " is from " + num(lo) + " to " + num(hi) + "; it is " + num(d) + ".", at);
        }
    }

    static std::string num(double d) {
        std::string s = std::to_string(d);
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
        return s;
    }

    void options(const Entity& e, const std::string& at) {
        const nlohmann::json& o = e.options;
        if (!o.is_object()) return;
        const std::string op = at + ".options";
        const std::vector<std::string>& known = entity_options(e.draw);
        for (auto it = o.begin(); it != o.end(); ++it) {
            const std::string key = it.key();
            const nlohmann::json& v = it.value();
            const std::string kp = key_path(op, key);
            if (!contains(known, key)) {
                error("spec.unknown_option",
                      "A " + std::string(draw_kind_name(e.draw)) + " entity has no option " + quote(key) +
                          "; its options are " + join(known) + ".",
                      kp);
                continue;
            }
            const bool ref = looks_like_reference(v);
            if (key == "colormap") {
                if (!ref && !(v.is_string() && colormap_id(v.get<std::string>())))
                    error("spec.bad_value", "colormap is one of " + join(colormap_names()) + ", or a reference.", kp);
            } else if (key == "vmin" || key == "vmax") {
                if (!ref && !v.is_number()) error("spec.bad_value", key + " is a number or a reference.", kp);
            } else if (key == "size_px") {
                number_range(v, kp, 0.5, 64, "size_px");
            } else if (key == "width_px") {
                number_range(v, kp, 0.5, 32, "width_px");
            } else if (key == "shape") {
                if (!(v == "disc" || v == "square")) error("spec.bad_value", "shape is disc or square.", kp);
            } else if (key == "mode") {
                if (!(v == "strip" || v == "segments")) error("spec.bad_value", "mode is strip or segments.", kp);
            } else if (key == "x" || key == "y" || key == "z") {
                if (!(v.is_number() || v == "min" || v == "max"))
                    error("spec.bad_value", key + " is a number in data units, or min or max of the data box.", kp);
            } else if (key == "own") {
                std::set<std::string> axes;
                bool ok = v.is_array() && v.size() <= 3;
                if (ok)
                    for (const auto& a : v)
                        ok = ok && (a == "x" || a == "y" || a == "z") && axes.insert(a.get<std::string>()).second;
                if (!ok) error("spec.bad_value", "own is a list of distinct axes among x, y and z.", kp);
            } else if (key == "depth_offset") {
                if (!v.is_number_integer())
                    error("spec.bad_value", "depth_offset is an integer from -8 to 8.", kp);
                else
                    number_range(v, kp, -8, 8, "depth_offset");
            } else if (key == "highlight") {
                highlight(e, v, kp);
            } else if (key == "visible") {
                if (!ref && !v.is_boolean()) error("spec.bad_value", "visible is true, false or a reference.", kp);
            }
        }
        if (o.contains("vmin") && o.contains("vmax") && o["vmin"].is_number() && o["vmax"].is_number() &&
            !(o["vmin"].get<double>() < o["vmax"].get<double>()))
            error("spec.out_of_range", "vmin is below vmax.", key_path(op, "vmin"));
    }

    void highlight(const Entity& e, const nlohmann::json& v, const std::string& kp) {
        if (!v.is_object()) {
            error("spec.bad_value", "highlight is a reference with the key along.", kp);
            return;
        }
        auto along = v.find("along");
        const std::vector<std::string> allowed = e.draw == DrawKind::surface
                                                     ? std::vector<std::string>{"row", "column"}
                                                     : std::vector<std::string>{"index"};
        if (along == v.end() || !along->is_string() || !contains(allowed, along->get<std::string>())) {
            error("spec.bad_value",
                  "highlight of a " + std::string(draw_kind_name(e.draw)) + " entity takes along = " + join(allowed) +
                      ".",
                  key_path(kp, "along"));
            return;
        }
        nlohmann::json ref = v;
        ref.erase("along");
        if (!parse_reference(ref))
            error("spec.bad_reference", "highlight names one control, selection or state, with along.", kp);
    }

    void pose(const Entity& e, const std::string& at) {
        if (!e.pose) return;
        const Pose& p = *e.pose;
        const std::string pp = at + ".pose";
        const auto& q = p.rotation;
        if (!(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] > 1e-12))
            error("spec.bad_value", "A pose's rotation is a quaternion [x, y, z, w] that is not zero.",
                  pp + ".rotation");
        for (double s : p.scale)
            if (!(std::fabs(s) > 0)) {
                error("spec.out_of_range", "Each scale of a pose is not zero.", pp + ".scale");
                break;
            }
    }

    void box(const Panel& p, const std::string& path) {
        const nlohmann::json& b = p.options.box;
        if (b.is_null()) return;
        const std::string bp = path + ".options.box";
        if (p.primitive != Primitive::scene) {
            error("spec.unknown_option", "Only a scene panel has the option box.", bp);
            return;
        }
        if (b == "auto") return;
        bool ok = b.is_array() && b.size() == 3;
        if (ok)
            for (const auto& v : b) ok = ok && v.is_number() && v.get<double>() > 0 && std::isfinite(v.get<double>());
        if (!ok) error("spec.bad_value", "box is auto, or three positive numbers: the proportions of the box.", bp);
    }

    void views(const Panel& p, const std::string& path) {
        if (p.options.views.empty()) return;
        const std::string vp = path + ".options.views";
        if (p.primitive != Primitive::scene) {
            error("spec.unknown_option", "Only a scene panel has the option views.", vp);
            return;
        }
        if (p.options.views.size() > limits::kSavedViews) {
            error("spec.too_many", "A scene panel keeps at most " + std::to_string(limits::kSavedViews) +
                                       " saved views.",
                  vp);
            return;
        }
        std::set<std::string> names;
        for (std::size_t k = 0; k < p.options.views.size(); ++k) {
            const SavedView& v = p.options.views[k];
            const std::string at = index_path(vp, k);
            if (v.name.empty() || code_points(v.name) > 64 || has_control(v.name, TextKind::line))
                error("spec.bad_value", "A saved view's name is 1 to 64 characters on one line.", at + ".name");
            else if (!names.insert(v.name).second)
                error("spec.duplicate_id", "Two saved views of " + quote(p.id) + " have the name " + quote(v.name) + ".",
                      at + ".name");
            if (!(v.distance > 0) || !std::isfinite(v.distance))
                error("spec.out_of_range", "A saved view's distance is above 0.", at + ".distance");
            if (!(v.elevation >= -89 && v.elevation <= 89))
                error("spec.out_of_range", "A saved view's elevation is from -89 to 89 degrees.", at + ".elevation");
            if (!std::isfinite(v.azimuth)) error("spec.bad_value", "A saved view's azimuth is finite.", at + ".azimuth");
        }
    }

    const AdditionsCheck& c_;
};

}  // namespace

void validate_entities(const AdditionsCheck& c) { EntityCheck(c).run(); }

}  // namespace caliper_builder::spec::detail
