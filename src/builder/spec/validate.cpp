// Validation of a spec: ids, lengths and ranges; references, cycles and the
// layout; slot rules; and shape propagation through `from` and the tensor
// transforms when a SourceResolver knows the catalog sources.
#include <caliper_builder/spec/validate.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>

#include "internal.h"

namespace caliper_builder::spec {

namespace {

using namespace detail;

const std::vector<std::string>& dtypes() {
    static const std::vector<std::string> v = {"u8",  "i8",  "u16", "i16", "u32", "i32",
                                               "u64", "i64", "f16", "f32", "f64", "bool"};
    return v;
}

// ---- column types -------------------------------------------------------------

enum class ColType { unknown, numeric, temporal, other };

ColType classify(const std::string& raw) {
    std::string t;
    for (char c : raw) t += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    while (!t.empty() && t.back() == ' ') t.pop_back();
    while (!t.empty() && t.front() == ' ') t.erase(t.begin());
    if (t.empty()) return ColType::unknown;
    if (t.back() == ']') return ColType::other;  // LIST and ARRAY types, such as FLOAT[] or FLOAT[3]
    std::string base = t.substr(0, t.find('('));
    while (!base.empty() && base.back() == ' ') base.pop_back();
    static const std::set<std::string> numeric = {
        "BOOLEAN", "BOOL",     "LOGICAL",  "TINYINT", "INT1",   "SMALLINT", "INT2",    "SHORT",   "INTEGER",
        "INT",     "INT4",     "SIGNED",   "BIGINT",  "INT8",   "LONG",     "HUGEINT", "INT128",  "UTINYINT",
        "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "FLOAT4",  "REAL",    "DOUBLE",  "FLOAT8",
        "DECIMAL", "NUMERIC",  "VARINT",   "BIGNUM"};
    static const std::set<std::string> temporal = {
        "DATE",         "TIME",         "TIMESTAMP",   "DATETIME",    "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP_S",  "TIMESTAMP_MS", "TIMESTAMP_NS", "TIMESTAMP_US", "TIMETZ",     "TIME WITH TIME ZONE"};
    static const std::set<std::string> other = {
        "VARCHAR", "CHAR", "BPCHAR", "TEXT",   "STRING", "BLOB", "BYTEA",  "BINARY", "VARBINARY", "UUID",
        "JSON",    "BIT",  "BITSTRING", "INTERVAL", "ENUM", "STRUCT", "MAP", "UNION", "LIST",  "ARRAY"};
    if (numeric.count(base)) return ColType::numeric;
    if (temporal.count(base)) return ColType::temporal;
    if (other.count(base)) return ColType::other;
    return ColType::unknown;
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += shape[i] < 0 ? std::string("?") : std::to_string(shape[i]);
    }
    return s + "]";
}

// Python's len(range(start, stop, step)) over an axis of size n, with step >= 1.
std::int64_t slice_length(const std::optional<std::int64_t>& start, const std::optional<std::int64_t>& stop,
                          std::int64_t step, std::int64_t n) {
    auto clamp = [n](const std::optional<std::int64_t>& v, std::int64_t fallback) {
        if (!v) return fallback;
        std::int64_t x = *v;
        if (x < 0) {
            x += n;
            if (x < 0) x = 0;
        } else if (x > n) {
            x = n;
        }
        return x;
    };
    const std::int64_t s = clamp(start, 0);
    const std::int64_t e = clamp(stop, n);
    if (e <= s) return 0;
    return (e - s + step - 1) / step;
}

// What one binding names.
struct Bound {
    enum class What { none, unknown, table, column, tensor };
    What what = What::none;
    int source = -1;
    std::string column;
    ColType type = ColType::unknown;
    std::string type_name;
    const SourceInfo* info = nullptr;
    std::string path;
    std::string text;

    bool bound() const { return what != What::none; }
    bool known_shape() const { return what == What::tensor && info && info->shape.has_value(); }
    const std::vector<std::int64_t>& shape() const { return *info->shape; }
};

class Validator {
public:
    Validator(const Spec& spec, const SourceResolver& resolver, ValidationReport& report)
        : spec_(spec), resolver_(resolver), report_(report), sink_(report.diagnostics) {}

    void run() {
        if (spec_.version < kFormatVersion || spec_.version > kLatestFormatVersion) {
            sink_.error("spec.unsupported_version",
                        "This spec has format version " + std::to_string(spec_.version) +
                            ", and this reader knows the versions " + std::to_string(kFormatVersion) + " to " +
                            std::to_string(kLatestFormatVersion) + ".",
                        "caliper_spec");
            return;
        }
        applet();
        version_keys();
        ids();
        const std::size_t n = spec_.sources.size();
        report_.sources.resize(n);
        state_.assign(n, 0);
        root_.assign(n, -1);
        form_ok_.assign(n, false);
        transforms_ok_.assign(n, {});
        used_.assign(n, false);
        for (std::size_t i = 0; i < n; ++i) source_local(i);
        for (std::size_t i = 0; i < n; ++i) resolve(static_cast<int>(i));
        for (std::size_t i = 0; i < spec_.panels.size(); ++i) panel(i);
        for (std::size_t i = 0; i < spec_.controls.size() && i < limits::kControls; ++i) control(i);
        layout();
        usage();
        extension(spec_.x, "x");
    }

private:
    // ---- small checks ----------------------------------------------------------

    void text(const std::string& s, const std::string& path, std::size_t limit, TextKind kind, bool allow_empty) {
        if (!valid_utf8(s)) {
            sink_.error("spec.bad_value", path + " is not valid UTF-8.", path);
            return;
        }
        if (code_points(s) > limit) {
            sink_.error("spec.string_too_long",
                        path + " is longer than " + std::to_string(limit) + " characters.", path);
            return;
        }
        if (has_control(s, kind)) {
            sink_.error("spec.bad_value",
                        path + (kind == TextKind::line ? " holds a control character."
                                                       : " holds a control character other than tab or a line break."),
                        path);
            return;
        }
        if (!allow_empty && s.empty()) sink_.error("spec.bad_value", path + " is empty.", path);
    }

    void id(const std::string& value, const std::string& path) {
        if (valid_id(value)) return;
        sink_.error("spec.bad_id",
                    (value.empty() ? path + " is empty" : quote(value) + " is not a valid id") +
                        "; an id is a lower-case letter followed by at most 63 lower-case letters, digits, _ or -.",
                    path);
    }

    void extension(const nlohmann::json& x, const std::string& path) {
        if (x.is_null()) return;
        if (!x.is_object()) {
            sink_.error("spec.wrong_type", path + " must be a table.", path);
            return;
        }
        int depth = 1;
        for (char c : path)
            if (c == '.' || c == '[') ++depth;
        check_values(x, path, depth, x_values_, sink_);
    }

    // ---- the applet and the ids -----------------------------------------------

    void applet() {
        id(spec_.applet.id, "applet.id");
        if (spec_.applet.title) text(*spec_.applet.title, "applet.title", limits::kTitleChars, TextKind::line, false);
        if (spec_.applet.description)
            text(*spec_.applet.description, "applet.description", limits::kDescriptionChars, TextKind::long_text,
                 true);
        extension(spec_.applet.x, "applet.x");
    }

    // A Spec of an older version that holds what a later version adds gets the
    // errors that its version's reader gives (design/instruments.md).
    void version_keys() {
        for (std::size_t i = 0; i < spec_.panels.size(); ++i) {
            const Primitive p = spec_.panels[i].primitive;
            if (primitive_version(p) > spec_.version)
                sink_.error("spec.unknown_primitive",
                            quote(to_string(p)) + " is not a primitive of format version " +
                                std::to_string(spec_.version) + "; raise caliper_spec to " +
                                std::to_string(primitive_version(p)) + ".",
                            index_path("panel", i) + ".primitive");
        }
        if (!spec_.controls.empty() && spec_.version < kInstrumentVersion)
            sink_.error("spec.unknown_key",
                        "\"control\" is not a key of a spec of format version " + std::to_string(spec_.version) +
                            "; raise caliper_spec to " + std::to_string(kInstrumentVersion) + ".",
                        "control");
    }

    void ids() {
        if (spec_.sources.size() > limits::kSources)
            sink_.error("spec.too_many",
                        "The spec has " + std::to_string(spec_.sources.size()) + " sources; the limit is " +
                            std::to_string(limits::kSources) + ".",
                        "source");
        if (spec_.panels.size() > limits::kPanels)
            sink_.error("spec.too_many",
                        "The spec has " + std::to_string(spec_.panels.size()) + " panels; the limit is " +
                            std::to_string(limits::kPanels) + ".",
                        "panel");
        for (std::size_t i = 0; i < spec_.sources.size(); ++i) {
            const std::string path = index_path("source", i) + ".id";
            const std::string& v = spec_.sources[i].id;
            id(v, path);
            if (!sources_.emplace(v, static_cast<int>(i)).second)
                sink_.error("spec.duplicate_id", "Two sources have the id " + quote(v) + ".", path);
        }
        for (std::size_t i = 0; i < spec_.panels.size(); ++i) {
            const std::string path = index_path("panel", i) + ".id";
            const std::string& v = spec_.panels[i].id;
            id(v, path);
            if (!panels_.emplace(v, static_cast<int>(i)).second)
                sink_.error("spec.duplicate_id", "Two panels have the id " + quote(v) + ".", path);
        }
        if (spec_.controls.size() > limits::kControls)
            sink_.error("spec.too_many",
                        "The spec has " + std::to_string(spec_.controls.size()) + " controls; the limit is " +
                            std::to_string(limits::kControls) + ".",
                        "control");
        for (std::size_t i = 0; i < spec_.controls.size(); ++i) {
            const std::string path = index_path("control", i) + ".id";
            const std::string& v = spec_.controls[i].id;
            id(v, path);
            if (panels_.count(v)) {
                sink_.error("spec.duplicate_id",
                            "The control id " + quote(v) + " is also a panel id; panels and controls share one set "
                            "of ids.",
                            path);
            } else if (!controls_.emplace(v, static_cast<int>(i)).second) {
                sink_.error("spec.duplicate_id", "Two controls have the id " + quote(v) + ".", path);
            }
        }
    }

    // ---- controls (spec version 1) -----------------------------------------------

    void control(std::size_t i) {
        const Control& c = spec_.controls[i];
        const std::string path = index_path("control", i);
        const std::string kind = control_kind_name(c.kind);
        if (c.title) text(*c.title, path + ".title", limits::kTitleChars, TextKind::line, false);
        extension(c.x, path + ".x");
        extension(c.options.x, path + ".options.x");
        const std::optional<ControlTarget> target = parse_control_target(c.bind);
        if (!target) {
            if (c.bind.rfind("intervene:", 0) == 0) {
                sink_.error("spec.bad_binding",
                            "The target " + quote(c.bind) + " is reserved for a later Caliper; a control binds " +
                                "param:<name>, action:<name>, run:pause or value.",
                            path + ".bind");
            } else {
                sink_.error("spec.bad_binding",
                            quote(c.bind) + " is not a target; a control binds param:<name>, action:<name>, " +
                                "run:pause or value, with a name of 1 to 63 bytes.",
                            path + ".bind");
            }
        }
        if (!control_kind_built(c.kind)) {
            sink_.error("spec.unsupported_control",
                        "This Caliper does not draw a " + kind + " control yet; the kinds it draws are slider and "
                        "button.",
                        path + ".kind");
            return;
        }
        if (target && !control_takes_target(c.kind, *target))
            sink_.error("spec.control_mismatch",
                        "A " + kind + " cannot bind " + quote(c.bind) + "; a " + kind + " binds " +
                            (c.kind == ControlKind::slider ? "param:<name> or value." : "action:<name> or run:pause."),
                        path + ".bind");
        const ControlOptions& o = c.options;
        const std::string op = path + ".options";
        if (c.kind == ControlKind::button) {
            using Set = std::pair<const char*, bool>;
            const Set set[] = {{"log", o.log.has_value()},
                               {"max", o.max.has_value()},
                               {"min", o.min.has_value()},
                               {"step", o.step.has_value()},
                               {"value", o.value.has_value()}};
            for (const Set& opt : set)
                if (opt.second)
                    sink_.error("spec.control_mismatch",
                                quote(opt.first) + " is an option of a slider; a button takes no options.",
                                op + "." + opt.first);
            return;
        }
        // A slider.
        if (o.min && o.max && !(*o.min < *o.max))
            sink_.error("spec.out_of_range", op + ".min must be below " + op + ".max.", op + ".min");
        if (o.step && !(*o.step > 0))
            sink_.error("spec.out_of_range", op + ".step must be above 0.", op + ".step");
        if (o.log && *o.log && o.min && !(*o.min > 0))
            sink_.error("spec.out_of_range", "A log slider needs " + op + ".min above 0.", op + ".min");
    }

    // ---- scene panels (spec version 1) -------------------------------------------

    void scene_panel(const Panel& p, const std::string& path) {
        for (const auto& [slot, value] : p.bind) {
            const std::string sp = path + ".bind." + slot;
            if (slot != kSceneSlot) {
                sink_.error("spec.unknown_slot", quote(slot) + " is not a slot of a scene; its slots are \"scene\".",
                            sp);
                continue;
            }
            if (value.bindings.empty()) {
                sink_.error("spec.missing_slot", sp + " holds no binding.", sp);
                continue;
            }
            if (value.list || value.bindings.size() > 1) {
                sink_.error("spec.wrong_type", sp + " takes one binding, not a list.", sp);
                continue;
            }
            const std::string& binding = value.bindings[0];
            if (valid_utf8(binding) && code_points(binding) > 256) {
                sink_.error("spec.string_too_long", sp + " holds a binding longer than 256 characters.", sp);
                continue;
            }
            const std::optional<std::string> name = live_scene_name(binding);
            if (!name) {
                sink_.error("spec.bad_binding",
                            quote(binding) + " is not a live scene; the slot scene holds live:<scene name>, as "
                            "catalog.list names the scene.",
                            sp);
                continue;
            }
            if (!resolver_) continue;
            std::optional<ResolvedSource> r;
            try {
                r = resolver_(binding);
            } catch (...) {
                sink_.error("spec.internal", "The source resolver failed for " + quote(binding) + ".", sp);
                continue;
            }
            if (!r)
                sink_.warning("spec.unresolved_source",
                              "The attached script does not send the scene " + quote(*name) +
                                  " now, so the panel shows that its source is offline until it does.",
                              sp);
        }
        if (!p.bind.count(kSceneSlot))
            sink_.error("spec.missing_slot", "A scene panel needs the slot \"scene\".", path + ".bind.scene");
    }

    void scene_options(const SceneOptions& o, const std::string& op) {
        if (o.view_camera.is_string()) {
            text(o.view_camera.get<std::string>(), op + ".view_camera", 255, TextKind::line, false);
        } else if (o.view_camera.is_object()) {
            const std::string cp = op + ".view_camera";
            if (auto d = o.view_camera.find("distance"); d != o.view_camera.end() && d->is_number() &&
                                                         !(d->get<double>() >= 0))
                sink_.error("spec.out_of_range", cp + ".distance must not be below 0.", cp + ".distance");
            if (auto e = o.view_camera.find("elevation"); e != o.view_camera.end() && e->is_number()) {
                const double v = e->get<double>();
                if (!(v >= -89 && v <= 89))
                    sink_.error("spec.out_of_range", cp + ".elevation must be from -89 to 89 degrees.",
                                cp + ".elevation");
            }
        }
        using List = std::pair<const char*, const std::vector<std::string>*>;
        const List lists[] = {{"include", &o.include}, {"exclude", &o.exclude}};
        for (const List& l : lists) {
            const std::string lp = op + "." + l.first;
            if (l.second->size() > limits::kScenePatterns) {
                sink_.error("spec.too_many",
                            lp + " lists " + std::to_string(l.second->size()) + " patterns; the limit is " +
                                std::to_string(limits::kScenePatterns) + ".",
                            lp);
                continue;
            }
            for (std::size_t k = 0; k < l.second->size(); ++k)
                text((*l.second)[k], index_path(lp, k), 255, TextKind::line, false);
        }
    }

    // ---- sources ---------------------------------------------------------------

    void source_local(std::size_t i) {
        const Source& s = spec_.sources[i];
        const std::string path = index_path("source", i);
        const int forms = (s.catalog ? 1 : 0) + (s.sql ? 1 : 0) + (s.from ? 1 : 0);
        if (forms != 1) {
            sink_.error("spec.source_form",
                        "The source " + quote(s.id) +
                            (forms == 0 ? " has none of catalog, sql and from; it needs exactly one."
                                        : " has more than one of catalog, sql and from; it needs exactly one."),
                        path);
        }
        bool ok = forms == 1;
        if (s.catalog) {
            const std::string cp = path + ".catalog";
            const std::size_t before = sink_.errors();
            text(*s.catalog, cp, limits::kCatalogChars, TextKind::line, false);
            if (sink_.errors() == before) {
                const std::size_t colon = s.catalog->find(':');
                const bool good = colon != std::string::npos && colon + 1 < s.catalog->size() &&
                                  contains(catalog_kinds(), s.catalog->substr(0, colon));
                if (!good)
                    sink_.error("spec.bad_value",
                                quote(*s.catalog) +
                                    " is not a catalog id; a catalog id is <kind>:<name>, with a kind of " +
                                    join(catalog_kinds()) + ".",
                                cp);
            }
            ok = ok && sink_.errors() == before;
        }
        if (s.sql) text(*s.sql, path + ".sql", limits::kSqlChars, TextKind::long_text, false);
        if (s.from && sources_.find(*s.from) == sources_.end()) {
            sink_.error("spec.unknown_source",
                        "The source " + quote(s.id) + " derives from " + quote(*s.from) +
                            ", which is not a source of this spec.",
                        path + ".from");
        }
        form_ok_[i] = ok;
        if (s.transforms.size() > limits::kTransforms)
            sink_.error("spec.too_many",
                        "The source " + quote(s.id) + " has " + std::to_string(s.transforms.size()) +
                            " transforms; the limit is " + std::to_string(limits::kTransforms) + ".",
                        path + ".transform");
        for (std::size_t j = 0; j < s.transforms.size(); ++j)
            transforms_ok_[i].push_back(transform_local(s.transforms[j], index_path(path + ".transform", j)));
        extension(s.x, path + ".x");
    }

    bool transform_local(const Transform& t, const std::string& path) {
        const std::size_t before = sink_.errors();
        const auto& allowed = transform_keys(t.op);
        const std::string owner = std::string("a ") + to_string(t.op) + " transform";
        auto field = [&](bool set, const char* key) {
            if (set && !contains(allowed, key))
                sink_.error("spec.unknown_key", quote(key) + " is not a key of " + owner + ".", path + "." + key);
        };
        field(!t.axis.empty(), "axis");
        field(t.index.has_value(), "index");
        field(t.start.has_value(), "start");
        field(t.stop.has_value(), "stop");
        field(t.step.has_value(), "step");
        field(t.fn.has_value(), "fn");
        field(t.keepdims.has_value(), "keepdims");
        field(t.mode.has_value(), "mode");

        const std::string ap = path + ".axis";
        auto axis_range = [&](std::int64_t a, const std::string& p) {
            if (a < -limits::kMaxRank || a >= limits::kMaxRank)
                sink_.error("spec.out_of_range",
                            p + " is " + std::to_string(a) + "; an axis is from -8 to 7, since a tensor has at most 8 "
                                "dimensions.",
                            p);
        };
        auto safe = [&](const std::optional<std::int64_t>& v, const char* key) {
            if (v && (*v > limits::kMaxSafeInteger || *v < -limits::kMaxSafeInteger))
                sink_.error("spec.out_of_range", path + "." + key + " is outside +-(2^53 - 1).", path + "." + key);
        };
        switch (t.op) {
            case TransformOp::slice:
                if (t.axis.empty()) {
                    sink_.error("spec.missing_key", "A slice transform needs the key \"axis\".", ap);
                } else if (t.axis_list || t.axis.size() > 1) {
                    sink_.error("spec.wrong_type", ap + " must be one integer.", ap);
                } else {
                    axis_range(t.axis[0], ap);
                }
                if (t.index && (t.start || t.stop || t.step))
                    sink_.error("spec.bad_transform",
                                "A slice has either index or a range of start, stop and step, not both.", path);
                safe(t.index, "index");
                safe(t.start, "start");
                safe(t.stop, "stop");
                if (t.step && (*t.step < 1 || *t.step > limits::kMaxSafeInteger))
                    sink_.error("spec.out_of_range",
                                path + ".step is " + std::to_string(*t.step) + "; a step is at least 1.",
                                path + ".step");
                break;
            case TransformOp::reduce: {
                if (t.axis.empty()) {
                    sink_.error("spec.missing_key", "A reduce transform needs the key \"axis\".", ap);
                } else if (t.axis.size() > limits::kReduceAxes) {
                    sink_.error("spec.too_many", ap + " lists more than 8 axes.", ap);
                } else {
                    std::set<std::int64_t> seen;
                    for (std::size_t k = 0; k < t.axis.size(); ++k) {
                        const std::string p = t.axis_list ? index_path(ap, k) : ap;
                        axis_range(t.axis[k], p);
                        if (!seen.insert(t.axis[k]).second)
                            sink_.error("spec.bad_transform", ap + " names the axis " + std::to_string(t.axis[k]) +
                                                                  " twice.",
                                        p);
                    }
                }
                if (!t.fn) {
                    sink_.error("spec.missing_key", "A reduce transform needs the key \"fn\".", path + ".fn");
                } else if (!contains(reduce_fns(), *t.fn)) {
                    sink_.error("spec.bad_value",
                                quote(*t.fn) + " is not a reduction; the choices are " + join(reduce_fns()) + ".",
                                path + ".fn");
                } else if ((*t.fn == "argmax" || *t.fn == "argmin") && t.axis.size() > 1) {
                    sink_.error("spec.bad_transform", *t.fn + " reduces over one axis only.", ap);
                }
                break;
            }
            case TransformOp::normalise:
                if (!t.mode) {
                    sink_.error("spec.missing_key", "A normalise transform needs the key \"mode\".", path + ".mode");
                } else if (!contains(normalise_modes(), *t.mode)) {
                    sink_.error("spec.bad_value",
                                quote(*t.mode) + " is not a normalise mode; the choices are " +
                                    join(normalise_modes()) + ".",
                                path + ".mode");
                }
                if (t.axis_list || t.axis.size() > 1) {
                    sink_.error("spec.wrong_type", ap + " must be one integer.", ap);
                } else if (!t.axis.empty()) {
                    axis_range(t.axis[0], ap);
                }
                break;
        }
        extension(t.x, path + ".x");
        return sink_.errors() == before;
    }

    // Resolves the sources in the chain of `from` that ends at source i, from
    // its origin forward, without recursion.
    void resolve(int i) {
        if (state_[i] == 2) return;
        std::vector<int> chain;
        int cur = i;
        bool cycle = false;
        while (true) {
            if (state_[cur] == 2) break;
            if (state_[cur] == 1) {
                cycle = true;
                break;
            }
            state_[cur] = 1;
            chain.push_back(cur);
            const Source& s = spec_.sources[cur];
            if (!form_ok_[cur] || !s.from) break;
            auto it = sources_.find(*s.from);
            if (it == sources_.end()) break;
            cur = it->second;
        }
        if (cycle) {
            std::string names;
            bool in_cycle = false;
            for (int k : chain) {
                if (k == cur) in_cycle = true;
                if (in_cycle) names += quote(spec_.sources[k].id) + " -> ";
            }
            names += quote(spec_.sources[cur].id);
            sink_.error("spec.cycle", "The sources derive from each other in a cycle: " + names + ".",
                        index_path("source", static_cast<std::size_t>(cur)) + ".from");
            for (int k : chain) {
                report_.sources[k] = SourceInfo{};
                report_.sources[k].id = spec_.sources[k].id;
                state_[k] = 2;
            }
            return;
        }
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            const int k = *it;
            const Source& s = spec_.sources[k];
            SourceInfo info;
            if (form_ok_[k]) {
                if (s.catalog) {
                    info = from_catalog(k);
                    root_[k] = k;
                } else if (s.sql) {
                    info.kind = DataKind::tabular;
                } else if (s.from) {
                    auto f = sources_.find(*s.from);
                    if (f != sources_.end() && state_[f->second] == 2) {
                        info = report_.sources[f->second];
                        root_[k] = root_[f->second];
                    }
                }
            }
            info.id = s.id;
            apply_transforms(static_cast<std::size_t>(k), info);
            report_.sources[k] = std::move(info);
            state_[k] = 2;
        }
    }

    SourceInfo from_catalog(int k) {
        SourceInfo info;
        const std::string& cat = *spec_.sources[k].catalog;
        const std::string kind = cat.substr(0, cat.find(':'));
        const bool tabular = kind == "table" || kind == "feed" || kind == "metric";
        info.kind = tabular ? DataKind::tabular : DataKind::tensor;
        if (!resolver_) return info;
        const std::string path = index_path("source", static_cast<std::size_t>(k)) + ".catalog";
        std::optional<ResolvedSource> r;
        try {
            r = resolver_(cat);
        } catch (...) {
            sink_.error("spec.internal", "The source resolver failed for " + quote(cat) + ".", path);
            return info;
        }
        if (!r) {
            sink_.warning("spec.unresolved_source",
                          "The catalog does not know " + quote(cat) + ", so its shape is not checked.", path);
            return info;
        }
        info.wire = r->wire;
        const bool r_tabular = r->kind == "table" || r->kind == "feed" || r->kind == "metric";
        const bool r_tensor = r->kind == "tensor" || r->kind == "live";
        if ((tabular && !r_tabular) || (!tabular && !r_tensor)) return info;  // inconsistent: trust only the id
        if (tabular) {
            info.columns = r->columns;
        } else {
            if (contains(dtypes(), r->dtype)) info.dtype = r->dtype;
            if (r->shape && r->shape->size() <= static_cast<std::size_t>(limits::kMaxRank)) {
                bool good = true;
                for (std::int64_t d : *r->shape)
                    if (d < -1 || d > (std::int64_t{1} << 62)) good = false;
                if (good) info.shape = r->shape;
            }
        }
        return info;
    }

    void apply_transforms(std::size_t k, SourceInfo& info) {
        const Source& s = spec_.sources[k];
        for (std::size_t j = 0; j < s.transforms.size(); ++j) {
            const std::string path = index_path(index_path("source", k) + ".transform", j);
            if (info.kind == DataKind::tabular) {
                sink_.error("spec.bad_transform",
                            "Transforms apply only to tensors, and the source " + quote(s.id) + " is tabular.", path);
                return;
            }
            if (info.kind == DataKind::unknown) continue;
            if (!transforms_ok_[k][j]) {
                info.shape.reset();
                info.dtype.clear();
                continue;
            }
            if (!apply(s.transforms[j], info, path)) {
                info.shape.reset();
                info.dtype.clear();
            }
        }
    }

    // Applies one transform to a tensor's dtype and shape; false when it does
    // not apply to the shape.
    bool apply(const Transform& t, SourceInfo& info, const std::string& path) {
        auto float_dtype = [&] {
            if (info.dtype.empty()) return std::string();
            return info.dtype == "f64" ? std::string("f64") : std::string("f32");
        };
        auto norm_axis = [&](std::int64_t a, std::int64_t rank, const std::string& p, std::int64_t& out) {
            if (a < -rank || a >= rank) {
                sink_.error("spec.bad_transform",
                            "The axis " + std::to_string(a) + " is out of range for a tensor of shape " +
                                shape_text(*info.shape) + ".",
                            p);
                return false;
            }
            out = a < 0 ? a + rank : a;
            return true;
        };
        switch (t.op) {
            case TransformOp::slice: {
                if (!info.shape) return true;
                std::vector<std::int64_t>& shape = *info.shape;
                std::int64_t a = 0;
                if (!norm_axis(t.axis[0], static_cast<std::int64_t>(shape.size()), path + ".axis", a)) return false;
                const std::int64_t n = shape[a];
                if (t.index) {
                    if (n >= 0 && (*t.index < -n || *t.index >= n)) {
                        sink_.error("spec.bad_transform",
                                    "The index " + std::to_string(*t.index) + " is out of range for an axis of size " +
                                        std::to_string(n) + ".",
                                    path + ".index");
                        return false;
                    }
                    shape.erase(shape.begin() + a);
                } else {
                    shape[a] = n < 0 ? -1 : slice_length(t.start, t.stop, t.step.value_or(1), n);
                }
                return true;
            }
            case TransformOp::reduce: {
                const bool arg = t.fn && (*t.fn == "argmax" || *t.fn == "argmin");
                if (info.shape) {
                    std::vector<std::int64_t>& shape = *info.shape;
                    const auto rank = static_cast<std::int64_t>(shape.size());
                    std::vector<bool> reduced(shape.size(), false);
                    for (std::size_t k = 0; k < t.axis.size(); ++k) {
                        const std::string p = t.axis_list ? index_path(path + ".axis", k) : path + ".axis";
                        std::int64_t a = 0;
                        if (!norm_axis(t.axis[k], rank, p, a)) return false;
                        if (reduced[a]) {
                            sink_.error("spec.bad_transform",
                                        "The axes of this reduce name the axis " + std::to_string(a) + " twice.", p);
                            return false;
                        }
                        reduced[a] = true;
                    }
                    std::vector<std::int64_t> out;
                    for (std::size_t d = 0; d < shape.size(); ++d) {
                        if (!reduced[d]) {
                            out.push_back(shape[d]);
                        } else if (t.keepdims.value_or(false)) {
                            out.push_back(1);
                        }
                    }
                    shape = std::move(out);
                }
                info.dtype = arg ? std::string("i64") : float_dtype();
                return true;
            }
            case TransformOp::normalise: {
                if (info.shape && !t.axis.empty()) {
                    std::int64_t a = 0;
                    if (!norm_axis(t.axis[0], static_cast<std::int64_t>(info.shape->size()), path + ".axis", a))
                        return false;
                }
                info.dtype = float_dtype();
                return true;
            }
        }
        return true;
    }

    // ---- panels ----------------------------------------------------------------

    Bound bind(const std::string& binding, const std::string& path) {
        Bound b;
        b.path = path;
        b.text = binding;
        const std::size_t dot = binding.find('.');
        const std::string src = binding.substr(0, dot);
        std::string column = dot == std::string::npos ? std::string() : binding.substr(dot + 1);
        if (!valid_utf8(binding) || !valid_id(src) ||
            (dot != std::string::npos && (column.empty() || has_control(column, TextKind::line)))) {
            sink_.error("spec.bad_binding",
                        quote(binding) + " is not a binding; a binding is <source> or <source>.<column>.", path);
            return b;
        }
        if (dot != std::string::npos && code_points(column) > limits::kColumnChars) {
            sink_.error("spec.string_too_long",
                        "The column name in " + path + " is longer than " + std::to_string(limits::kColumnChars) +
                            " characters.",
                        path);
            return b;
        }
        auto it = sources_.find(src);
        if (it == sources_.end()) {
            sink_.error("spec.unknown_source",
                        "The binding " + quote(binding) + " names " + quote(src) +
                            ", which is not a source of this spec.",
                        path);
            return b;
        }
        b.source = it->second;
        b.info = &report_.sources[it->second];
        used_[it->second] = true;
        if (dot == std::string::npos) {
            b.what = b.info->kind == DataKind::tabular  ? Bound::What::table
                     : b.info->kind == DataKind::tensor ? Bound::What::tensor
                                                        : Bound::What::unknown;
            return b;
        }
        if (b.info->kind == DataKind::tensor) {
            sink_.error("spec.bad_binding",
                        "The binding " + quote(binding) + " names a column, but the source " + quote(src) +
                            " is a tensor.",
                        path);
            return b;
        }
        b.column = column;
        if (b.info->kind == DataKind::unknown) {
            b.what = Bound::What::unknown;
            return b;
        }
        b.what = Bound::What::column;
        if (b.info->columns) {
            const ResolvedColumn* found = nullptr;
            for (const ResolvedColumn& c : *b.info->columns)
                if (c.name == column) found = &c;
            if (!found) {
                std::vector<std::string> names;
                for (const ResolvedColumn& c : *b.info->columns) {
                    if (names.size() == 12) {
                        names.emplace_back("others");
                        break;
                    }
                    names.push_back(quote(c.name));
                }
                sink_.error("spec.unknown_column",
                            "The source " + quote(src) + " has no column " + quote(column) +
                                (names.empty() ? std::string(".") : "; its columns are " + join(names) + "."),
                            path);
                b.what = Bound::What::none;
                return b;
            }
            b.type = classify(found->type);
            b.type_name = found->type;
        }
        return b;
    }

    void mismatch(const Bound& b, const std::string& message) { sink_.error("spec.slot_mismatch", message, b.path); }

    // A slot that takes a vector: a column, or a tensor of rank 1 (or up to max_rank).
    void vector_slot(const Bound& b, const char* slot, bool temporal_ok, std::size_t max_rank) {
        switch (b.what) {
            case Bound::What::table:
                mismatch(b, std::string(slot) + " needs a column or a tensor, but " + quote(b.text) +
                                " is a whole table; bind one of its columns as " + quote(b.text + ".<column>") + ".");
                break;
            case Bound::What::column:
                if (b.type == ColType::other || (b.type == ColType::temporal && !temporal_ok))
                    mismatch(b, std::string(slot) + " needs a numeric column, but " + quote(b.column) +
                                    " has the type " + b.type_name + ".");
                break;
            case Bound::What::tensor:
                if (b.known_shape() && (b.shape().empty() || b.shape().size() > max_rank))
                    mismatch(b, std::string(slot) + " needs a tensor of rank " +
                                    (max_rank == 1 ? std::string("1") : std::string("1 or 2")) + ", but " +
                                    quote(b.text) + " has the shape " + shape_text(b.shape()) + ".");
                break;
            default: break;
        }
    }

    // The length of a vector-like tensor along `axis` (0 the first, -1 the last), or -1.
    static std::int64_t length(const Bound& b, int axis) {
        if (!b.known_shape() || b.shape().empty()) return -1;
        return axis < 0 ? b.shape().back() : b.shape().front();
    }

    void panel(std::size_t i) {
        const Panel& p = spec_.panels[i];
        const std::string path = index_path("panel", i);
        const std::string prim = to_string(p.primitive);
        if (p.title) text(*p.title, path + ".title", limits::kTitleChars, TextKind::line, false);
        extension(p.x, path + ".x");
        if (p.primitive == Primitive::scene) {
            scene_panel(p, path);
            options(p, path);
            return;
        }

        // Slots: names, then the bindings each one holds.
        std::map<std::string, std::vector<Bound>> bound;
        for (const auto& [slot, value] : p.bind) {
            const std::string sp = path + ".bind." + slot;
            if (!contains(slots_of(p.primitive), slot)) {
                sink_.error("spec.unknown_slot",
                            quote(slot) + " is not a slot of a " + prim + "; its slots are " +
                                join(slots_of(p.primitive)) + ".",
                            sp);
                continue;
            }
            const bool list_ok = p.primitive == Primitive::series && slot == "y";
            if (value.bindings.empty()) {
                sink_.error("spec.missing_slot", sp + " holds no binding.", sp);
                continue;
            }
            if ((value.list || value.bindings.size() > 1) && !list_ok) {
                sink_.error("spec.wrong_type", sp + " takes one binding, not a list.", sp);
                continue;
            }
            if (value.bindings.size() > limits::kYBindings) {
                sink_.error("spec.too_many",
                            sp + " holds " + std::to_string(value.bindings.size()) + " bindings; the limit is " +
                                std::to_string(limits::kYBindings) + ".",
                            sp);
                continue;
            }
            std::vector<Bound>& list = bound[slot];
            for (std::size_t k = 0; k < value.bindings.size(); ++k) {
                if (code_points(value.bindings[k]) > 320 && valid_utf8(value.bindings[k])) {
                    sink_.error("spec.string_too_long", sp + " holds a binding longer than 320 characters.",
                                value.list ? index_path(sp, k) : sp);
                    continue;
                }
                Bound b = bind(value.bindings[k], value.list ? index_path(sp, k) : sp);
                if (b.bound()) list.push_back(b);
            }
        }
        // A required slot that is absent; a slot whose bindings all failed was reported already.
        auto has = [&](const char* slot) { return p.bind.count(slot) > 0; };
        auto one = [&](const char* slot) -> Bound {
            auto it = bound.find(slot);
            return it == bound.end() || it->second.empty() ? Bound{} : it->second.front();
        };
        auto require = [&](const char* slot) {
            if (!has(slot))
                sink_.error("spec.missing_slot", "A " + prim + " panel needs the slot " + quote(slot) + ".",
                            path + ".bind." + slot);
        };

        switch (p.primitive) {
            case Primitive::series: {
                require("y");
                const Bound x = one("x");
                if (x.bound()) vector_slot(x, "x", true, 1);
                for (const Bound& y : bound["y"]) {
                    vector_slot(y, "y", false, 2);
                    if (x.what == Bound::What::column) {
                        if ((y.what == Bound::What::column && y.source != x.source) ||
                            y.what == Bound::What::tensor || y.what == Bound::What::table)
                            mismatch(y, "When x is a column, every y is a column of the same source, " +
                                            quote(spec_.sources[x.source].id) + ".");
                    } else if (x.what == Bound::What::tensor) {
                        if (y.what == Bound::What::column || y.what == Bound::What::table) {
                            mismatch(y, "When x is a tensor, every y is a tensor.");
                        } else {
                            const std::int64_t n = length(x, 0), m = length(y, -1);
                            if (n >= 0 && m >= 0 && n != m)
                                mismatch(y, "y has " + std::to_string(m) + " values along its last axis, and x has " +
                                                std::to_string(n) + ".");
                        }
                    }
                }
                break;
            }
            case Primitive::heatmap:
            case Primitive::image: {
                require("values");
                const Bound v = one("values");
                const bool heat = p.primitive == Primitive::heatmap;
                const std::string need = heat ? "values needs a tensor of rank 2"
                                              : "values needs a tensor [H, W], [H, W, 3] or [H, W, 4]";
                if (v.what == Bound::What::table || v.what == Bound::What::column) {
                    mismatch(v, need + ", but " + quote(v.text) + " is tabular.");
                } else if (v.known_shape()) {
                    const auto& s = v.shape();
                    const bool fits = heat ? s.size() == 2
                                           : (s.size() == 2 || (s.size() == 3 && (s[2] == 3 || s[2] == 4 || s[2] < 0)));
                    if (!fits) mismatch(v, need + ", but " + quote(v.text) + " has the shape " + shape_text(s) + ".");
                }
                break;
            }
            case Primitive::points: points(p, path, one); break;
            case Primitive::histogram: {
                require("values");
                const Bound v = one("values");
                if (v.what == Bound::What::table) {
                    mismatch(v, "values needs a column or a tensor, but " + quote(v.text) + " is a whole table.");
                } else if (v.what == Bound::What::column && (v.type == ColType::other || v.type == ColType::temporal)) {
                    mismatch(v, "values needs a numeric column, but " + quote(v.column) + " has the type " +
                                    v.type_name + ".");
                }
                break;
            }
            case Primitive::table: {
                require("rows");
                const Bound r = one("rows");
                if (r.what == Bound::What::column) {
                    mismatch(r, "rows binds a whole table; write " + quote(spec_.sources[r.source].id) + ", not " +
                                    quote(r.text) + ".");
                } else if (r.what == Bound::What::tensor) {
                    mismatch(r, "rows needs a tabular source, but " + quote(r.text) + " is a tensor.");
                }
                if (p.options.columns && r.what == Bound::What::table && r.info->columns) {
                    const auto& cols = *p.options.columns;
                    for (std::size_t k = 0; k < cols.size() && k < limits::kColumns; ++k) {
                        bool found = false;
                        for (const ResolvedColumn& c : *r.info->columns) found = found || c.name == cols[k];
                        if (!found)
                            sink_.error("spec.unknown_column",
                                        "The source " + quote(spec_.sources[r.source].id) + " has no column " +
                                            quote(cols[k]) + ".",
                                        index_path(path + ".options.columns", k));
                    }
                }
                break;
            }
            case Primitive::scene: break;   // scene_panel() above
            case Primitive::text: {
                if (!has("value") && !p.options.text)
                    sink_.error("spec.missing_slot", "A text panel needs bind.value, options.text or both.",
                                path + ".bind");
                const Bound v = one("value");
                if (v.what == Bound::What::table) {
                    mismatch(v, "value needs a column or a tensor of one element, but " + quote(v.text) +
                                    " is a whole table.");
                } else if (v.known_shape()) {
                    bool known = true, one_element = true;
                    for (std::int64_t d : v.shape()) {
                        if (d < 0) known = false;
                        if (d != 1) one_element = false;
                    }
                    if (known && !one_element)
                        mismatch(v, "value needs a tensor of one element, but " + quote(v.text) + " has the shape " +
                                        shape_text(v.shape()) + ".");
                }
                break;
            }
        }
        options(p, path);
    }

    template <class One>
    void points(const Panel& p, const std::string& path, One one) {
        const bool has_pos = p.bind.count("positions") > 0;
        const bool has_xyz = p.bind.count("x") || p.bind.count("y") || p.bind.count("z");
        if (has_pos && has_xyz) {
            sink_.error("spec.slot_mismatch", "positions excludes x, y and z; bind one or the other.",
                        path + ".bind.positions");
            return;
        }
        if (!has_pos) {
            for (const char* s : {"x", "y"})
                if (!p.bind.count(s))
                    sink_.error("spec.missing_slot",
                                "A points panel needs the slots \"x\" and \"y\", or the slot \"positions\".",
                                path + ".bind." + s);
        }
        const Bound pos = one("positions");
        if (pos.what == Bound::What::table || pos.what == Bound::What::column) {
            mismatch(pos, "positions needs a tensor [n, 2] or [n, 3], but " + quote(pos.text) + " is tabular.");
        } else if (pos.known_shape()) {
            const auto& s = pos.shape();
            if (!(s.size() == 2 && (s[1] == 2 || s[1] == 3 || s[1] < 0)))
                mismatch(pos, "positions needs a tensor [n, 2] or [n, 3], but " + quote(pos.text) +
                                  " has the shape " + shape_text(s) + ".");
        }
        std::vector<std::pair<const char*, Bound>> vectors;
        for (const char* s : {"x", "y", "z", "color", "size"}) {
            const Bound b = one(s);
            if (!b.bound()) continue;
            vector_slot(b, s, false, 1);
            vectors.emplace_back(s, b);
        }
        // Every vector matches the first bound slot: columns of one source, or
        // tensors of one length.
        const Bound base = pos.bound() ? pos : (vectors.empty() ? Bound{} : vectors.front().second);
        if (base.what == Bound::What::column) {
            for (const auto& [slot, b] : vectors)
                if ((b.what == Bound::What::column && b.source != base.source) || b.what == Bound::What::tensor)
                    mismatch(b, std::string(slot) + " must be a column of the same source as the other slots, " +
                                    quote(spec_.sources[base.source].id) + ".");
        } else if (base.what == Bound::What::tensor) {
            const std::int64_t n = length(base, 0);
            for (const auto& [slot, b] : vectors) {
                if (b.what == Bound::What::column) {
                    mismatch(b, std::string(slot) + " must be a tensor like the other slots, not a column.");
                } else if (b.what == Bound::What::tensor) {
                    const std::int64_t m = length(b, 0);
                    if (n >= 0 && m >= 0 && n != m)
                        mismatch(b, std::string(slot) + " has " + std::to_string(m) + " values, and the points number " +
                                        std::to_string(n) + ".");
                }
            }
        }
    }

    void options(const Panel& p, const std::string& path) {
        const Options& o = p.options;
        const std::string op = path + ".options";
        const auto& allowed = options_of(p.primitive);
        for (const std::string& name : o.set_names())
            if (!contains(allowed, name))
                sink_.error("spec.unknown_option",
                            quote(name) + " is not an option of a " + to_string(p.primitive) + "; its options are " +
                                join(allowed) + ".",
                            op + "." + name);
        using Label = std::pair<const char*, const std::optional<std::string>*>;
        const Label labels[] = {{"x_label", &o.x_label}, {"y_label", &o.y_label}, {"z_label", &o.z_label}};
        for (const Label& l : labels)
            if (*l.second) text(**l.second, op + "." + l.first, limits::kTitleChars, TextKind::line, true);
        using Range = std::pair<const char*, const std::optional<std::array<double, 2>>*>;
        const Range ranges[] = {{"x_range", &o.x_range}, {"y_range", &o.y_range}, {"range", &o.range}};
        for (const Range& r : ranges) {
            if (!*r.second) continue;
            const double lo = (**r.second)[0], hi = (**r.second)[1];
            if (!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
                sink_.error("spec.out_of_range",
                            op + "." + r.first + " must be [min, max] with two finite numbers and min below max.",
                            op + "." + r.first);
        }
        if (o.colormap && !contains(colormaps(), *o.colormap))
            sink_.error("spec.bad_value",
                        quote(*o.colormap) + " is not a colormap; the choices are " + join(colormaps()) + ".",
                        op + ".colormap");
        if (o.bins && (*o.bins < 1 || *o.bins > limits::kMaxBins))
            sink_.error("spec.out_of_range",
                        op + ".bins is " + std::to_string(*o.bins) + "; it must be from 1 to " +
                            std::to_string(limits::kMaxBins) + ".",
                        op + ".bins");
        if (o.columns) {
            const auto& cols = *o.columns;
            const std::string cp = op + ".columns";
            if (cols.empty()) {
                sink_.error("spec.bad_value", cp + " lists no column; leave it out to show every column.", cp);
            } else if (cols.size() > limits::kColumns) {
                sink_.error("spec.too_many",
                            cp + " lists " + std::to_string(cols.size()) + " columns; the limit is " +
                                std::to_string(limits::kColumns) + ".",
                            cp);
            } else {
                std::set<std::string> seen;
                for (std::size_t k = 0; k < cols.size(); ++k) {
                    text(cols[k], index_path(cp, k), limits::kColumnChars, TextKind::line, false);
                    if (!seen.insert(cols[k]).second)
                        sink_.error("spec.bad_value", cp + " lists " + quote(cols[k]) + " twice.", index_path(cp, k));
                }
            }
        }
        if (o.text) text(*o.text, op + ".text", limits::kTextChars, TextKind::long_text, true);
        if (p.primitive == Primitive::scene) scene_options(o.scene, op);
        extension(o.x, op + ".x");
    }

    // ---- layout ----------------------------------------------------------------

    void layout() {
        if (!spec_.layout) return;
        placed_.assign(spec_.panels.size(), 0);
        placed_controls_.assign(spec_.controls.size(), 0);
        walk(*spec_.layout, "layout", 1);
        for (std::size_t i = 0; i < spec_.controls.size(); ++i) {
            auto it = controls_.find(spec_.controls[i].id);
            if (it == controls_.end() || it->second != static_cast<int>(i)) continue;
            if (placed_controls_[i] == 0)
                sink_.warning("spec.unplaced_panel",
                              "The layout does not place the control " + quote(spec_.controls[i].id) +
                                  ", so it is not drawn.",
                              index_path("control", i));
        }
        for (std::size_t i = 0; i < spec_.panels.size(); ++i) {
            auto it = panels_.find(spec_.panels[i].id);
            if (it == panels_.end() || it->second != static_cast<int>(i)) continue;
            if (placed_[i] == 0)
                sink_.warning("spec.unplaced_panel",
                              "The layout does not place the panel " + quote(spec_.panels[i].id) +
                                  ", so it is not drawn.",
                              index_path("panel", i));
        }
    }

    void walk(const LayoutNode& n, const std::string& path, int depth) {
        if (++layout_nodes_ > limits::kLayoutNodes) {
            if (layout_nodes_ == limits::kLayoutNodes + 1)
                sink_.error("spec.too_many",
                            "The layout has more than " + std::to_string(limits::kLayoutNodes) + " nodes.", "layout");
            return;
        }
        if (depth > limits::kLayoutDepth) {
            sink_.error("spec.too_deep",
                        "The layout nests more than " + std::to_string(limits::kLayoutDepth) + " nodes deep here.",
                        path);
            return;
        }
        if (n.title) text(*n.title, path + ".title", limits::kTitleChars, TextKind::line, false);
        extension(n.x, path + ".x");
        if (n.kind == LayoutNode::Kind::panel) {
            if (!n.children.empty() || n.ratio || n.title || !n.x.is_null())
                sink_.error("spec.bad_layout", path + " places a panel, so it has no children, ratio, title or x.",
                            path);
            auto it = panels_.find(n.panel);
            if (it == panels_.end()) {
                auto c = controls_.find(n.panel);
                if (c == controls_.end()) {
                    sink_.error("spec.unknown_panel",
                                "The layout places " + quote(n.panel) +
                                    ", which is neither a panel nor a control of this spec.",
                                path);
                } else if (++placed_controls_[c->second] == 2) {
                    sink_.error("spec.duplicate_placement",
                                "The layout places the control " + quote(n.panel) + " more than once.", path);
                }
            } else if (++placed_[it->second] == 2) {
                sink_.error("spec.duplicate_placement",
                            "The layout places the panel " + quote(n.panel) + " more than once.", path);
            }
            return;
        }
        const std::size_t count = n.children.size();
        const bool split = n.kind == LayoutNode::Kind::split;
        const std::string list = path + (split ? ".children" : ".tabs");
        if (split && count < 2) {
            sink_.error("spec.bad_layout",
                        "A split needs at least two children, and " + path + " has " + std::to_string(count) + ".",
                        list);
        } else if (!split && count == 0) {
            sink_.error("spec.bad_layout", "Tabs need at least one child, and " + path + " has none.", list);
        } else if (count > limits::kLayoutChildren) {
            sink_.error("spec.too_many",
                        path + " has " + std::to_string(count) + " children; the limit is " +
                            std::to_string(limits::kLayoutChildren) + ".",
                        list);
        }
        if (n.ratio) {
            if (!split || count != 2) {
                sink_.error("spec.bad_layout",
                            "ratio applies only to a split of exactly two children; a split of more shares its space "
                            "equally.",
                            path + ".ratio");
            } else if (!std::isfinite(*n.ratio) || *n.ratio < 0.05 || *n.ratio > 0.95) {
                sink_.error("spec.out_of_range", path + ".ratio must be from 0.05 to 0.95.", path + ".ratio");
            }
        }
        for (std::size_t k = 0; k < count && k < limits::kLayoutChildren; ++k)
            walk(n.children[k], index_path(list, k), depth + 1);
    }

    // ---- usage and portability -----------------------------------------------

    void usage() {
        const std::size_t n = spec_.sources.size();
        // A source a panel binds uses its whole chain of `from`.
        std::vector<bool> used(n, false);
        for (std::size_t i = 0; i < n; ++i) {
            if (!used_[i]) continue;
            int cur = static_cast<int>(i);
            while (cur >= 0 && !used[cur]) {
                used[cur] = true;
                const Source& s = spec_.sources[cur];
                if (!s.from) break;
                auto it = sources_.find(*s.from);
                cur = it == sources_.end() ? -1 : it->second;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            const Source& s = spec_.sources[i];
            auto it = sources_.find(s.id);
            if (it == sources_.end() || it->second != static_cast<int>(i)) continue;  // a duplicate id
            if (!used[i]) {
                sink_.warning("spec.unused_source",
                              "No panel binds the source " + quote(s.id) + ", directly or through from.",
                              index_path("source", i));
                continue;
            }
            const SourceInfo& info = report_.sources[i];
            if (root_[i] == static_cast<int>(i) && info.wire.has_value() && !*info.wire)
                sink_.warning("spec.not_portable",
                              "The catalog source " + quote(*s.catalog) +
                                  " has no wire form, so this spec runs in the app but not in the browser.",
                              index_path("source", i) + ".catalog");
        }
    }

    const Spec& spec_;
    const SourceResolver& resolver_;
    ValidationReport& report_;
    Sink sink_;
    std::map<std::string, int> sources_;  // id to the index of its first source
    std::map<std::string, int> panels_;
    std::map<std::string, int> controls_;  // control id to its index; a control that repeats a panel id is left out
    std::vector<int> state_;  // 0 not seen, 1 on the chain being resolved, 2 resolved
    std::vector<int> root_;   // the catalog source a source derives from, or -1
    std::vector<bool> form_ok_;
    std::vector<std::vector<bool>> transforms_ok_;
    std::vector<bool> used_;  // bound directly by a panel
    std::vector<int> placed_;
    std::vector<int> placed_controls_;
    std::size_t layout_nodes_ = 0;
    std::size_t x_values_ = 0;
};

}  // namespace

ValidationReport validate(const Spec& spec, const SourceResolver& resolver) {
    ValidationReport report;
    try {
        Validator(spec, resolver, report).run();
    } catch (...) {
        report.diagnostics.push_back({Severity::error, "spec.internal", "The spec library failed unexpectedly.", ""});
    }
    return report;
}

ValidationReport check_toml(std::string_view text, const SourceResolver& resolver) {
    ParseResult parsed = parse_toml(text);
    if (!parsed.ok()) {
        ValidationReport report;
        report.diagnostics = std::move(parsed.diagnostics);
        return report;
    }
    ValidationReport report = validate(parsed.spec, resolver);
    locate(report.diagnostics, parsed.positions);
    return report;
}

const char* to_string(DataKind kind) {
    switch (kind) {
        case DataKind::tabular: return "tabular";
        case DataKind::tensor: return "tensor";
        default: return "unknown";
    }
}

nlohmann::json to_json(const SourceInfo& info) {
    nlohmann::json j = {{"id", info.id}, {"kind", to_string(info.kind)}};
    if (info.columns) {
        nlohmann::json cols = nlohmann::json::array();
        for (const ResolvedColumn& c : *info.columns) cols.push_back({{"name", c.name}, {"type", c.type}});
        j["columns"] = std::move(cols);
    } else {
        j["columns"] = nullptr;
    }
    j["dtype"] = info.dtype.empty() ? nlohmann::json() : nlohmann::json(info.dtype);
    j["shape"] = info.shape ? nlohmann::json(*info.shape) : nlohmann::json();
    j["wire"] = info.wire ? nlohmann::json(*info.wire) : nlohmann::json();
    return j;
}

}  // namespace caliper_builder::spec
