// The canonical JSON form of a spec: decoding a JSON tree into a Spec (the
// one decoder, which parse_toml also uses after it turns TOML into JSON) and
// writing a Spec back as JSON.
#include <caliper_builder/spec/format.h>

#include <cmath>
#include <set>
#include <stdexcept>

#include "internal.h"

namespace caliper_builder::spec {

namespace detail {

namespace {

const char* type_name(const json& v) {
    switch (v.type()) {
        case json::value_t::null: return "null";
        case json::value_t::boolean: return "a boolean";
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: return "an integer";
        case json::value_t::number_float: return "a number";
        case json::value_t::string: return "a string";
        case json::value_t::array: return "an array";
        case json::value_t::object: return "a table";
        default: return "a value of another kind";
    }
}

bool as_integer(const json& v, std::int64_t& out) {
    if (v.is_number_integer()) {  // signed or unsigned
        if (v.is_number_unsigned()) {
            const std::uint64_t u = v.get<std::uint64_t>();
            if (u > static_cast<std::uint64_t>(limits::kMaxSafeInteger)) return false;
            out = static_cast<std::int64_t>(u);
        } else {
            out = v.get<std::int64_t>();
        }
        return true;
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (!std::isfinite(d) || std::floor(d) != d) return false;
        if (std::fabs(d) > static_cast<double>(limits::kMaxSafeInteger)) return false;
        out = static_cast<std::int64_t>(d);
        return true;
    }
    return false;
}

class Decoder {
public:
    // `version` is the format version whose keys and primitives the items may
    // use; a whole document sets it from its caliper_spec.
    explicit Decoder(Sink& sink, std::int64_t version = kLatestFormatVersion) : sink_(sink), version_(version) {}

    bool table(const json& v, const std::string& path, const std::string& what) {
        if (v.is_object()) return true;
        sink_.error("spec.wrong_type", what + " must be a table, not " + type_name(v) + ".", path);
        return false;
    }

    // Reports every key of `o` that is not in `allowed` and is not `x`.
    void keys(const json& o, const std::string& path, const std::vector<std::string>& allowed,
              const std::string& owner) {
        for (auto it = o.begin(); it != o.end(); ++it) {
            if (it.key() == "x" || contains(allowed, it.key())) continue;
            sink_.error("spec.unknown_key", quote(it.key()) + " is not a key of " + owner + ".",
                        key_path(path, it.key()));
        }
    }

    void missing(const std::string& path, std::string_view key, const std::string& owner) {
        sink_.error("spec.missing_key", owner + " needs the key " + quote(key) + ".", key_path(path, key));
    }

    void wrong(const std::string& path, const json& v, const std::string& expected) {
        sink_.error("spec.wrong_type", path + " must be " + expected + ", not " + type_name(v) + ".", path);
    }

    std::optional<std::string> string(const json& o, const char* key, const std::string& path) {
        auto it = o.find(key);
        if (it == o.end()) return std::nullopt;
        if (!it->is_string()) {
            wrong(key_path(path, key), *it, "a string");
            return std::nullopt;
        }
        return it->get<std::string>();
    }

    std::optional<std::int64_t> integer(const json& o, const char* key, const std::string& path) {
        auto it = o.find(key);
        if (it == o.end()) return std::nullopt;
        std::int64_t v = 0;
        if (!as_integer(*it, v)) {
            wrong(key_path(path, key), *it, "an integer");
            return std::nullopt;
        }
        return v;
    }

    std::optional<double> number(const json& o, const char* key, const std::string& path) {
        auto it = o.find(key);
        if (it == o.end()) return std::nullopt;
        if (!it->is_number()) {
            wrong(key_path(path, key), *it, "a number");
            return std::nullopt;
        }
        return it->get<double>();
    }

    std::optional<bool> boolean(const json& o, const char* key, const std::string& path) {
        auto it = o.find(key);
        if (it == o.end()) return std::nullopt;
        if (!it->is_boolean()) {
            wrong(key_path(path, key), *it, "true or false");
            return std::nullopt;
        }
        return it->get<bool>();
    }

    std::optional<std::string> enumerated(const json& o, const char* key, const std::string& path,
                                          const std::vector<std::string>& values, const char* what) {
        auto s = string(o, key, path);
        if (s && !contains(values, *s)) {
            sink_.error("spec.bad_value", quote(*s) + " is not " + what + "; the choices are " + join(values) + ".",
                        key_path(path, key));
            return std::nullopt;
        }
        return s;
    }

    std::optional<std::array<double, 2>> range(const json& o, const char* key, const std::string& path) {
        auto it = o.find(key);
        if (it == o.end()) return std::nullopt;
        if (!it->is_array() || it->size() != 2 || !(*it)[0].is_number() || !(*it)[1].is_number()) {
            sink_.error("spec.wrong_type", key_path(path, key) + " must be [min, max], two numbers.",
                        key_path(path, key));
            return std::nullopt;
        }
        return std::array<double, 2>{(*it)[0].get<double>(), (*it)[1].get<double>()};
    }

    json extension(const json& o, const std::string& path) {
        auto it = o.find("x");
        if (it == o.end()) return json();
        if (!it->is_object()) {
            wrong(key_path(path, "x"), *it, "a table");
            return json();
        }
        if (it->empty()) return json();
        return *it;
    }

    Spec spec(const json& root) {
        Spec s;
        if (!root.is_object()) {
            sink_.error("spec.wrong_type", std::string("A spec must be a table, not ") + type_name(root) + ".", "");
            return s;
        }
        auto version = root.find("caliper_spec");
        if (version == root.end()) {
            missing("", "caliper_spec", "A spec");
        } else {
            std::int64_t v = 0;
            if (!as_integer(*version, v)) {
                wrong("caliper_spec", *version, "an integer");
            } else if (v < kFormatVersion || v > kLatestFormatVersion) {
                sink_.error("spec.unsupported_version",
                            "This spec has format version " + std::to_string(v) +
                                ", and this reader knows the versions " + std::to_string(kFormatVersion) + " to " +
                                std::to_string(kLatestFormatVersion) +
                                (v > kLatestFormatVersion ? "; open it with a newer Caliper." : "."),
                            "caliper_spec");
                return s;
            } else {
                s.version = v;
            }
        }
        version_ = s.version;
        if (version_ >= kInstrumentVersion) {
            keys(root, "",
                 {"caliper_spec", "applet", "source", "panel", "control", "program", "stage", "provenance", "layout"},
                 "a spec");
        } else {
            keys(root, "", {"caliper_spec", "applet", "source", "panel", "layout"}, "a spec");
        }

        auto applet = root.find("applet");
        if (applet == root.end()) {
            missing("", "applet", "A spec");
        } else if (table(*applet, "applet", "applet")) {
            keys(*applet, "applet", {"id", "title", "description"}, "the applet table");
            auto id = string(*applet, "id", "applet");
            if (id) {
                s.applet.id = *id;
            } else if (applet->find("id") == applet->end()) {
                missing("applet", "id", "The applet table");
            }
            s.applet.title = string(*applet, "title", "applet");
            s.applet.description = string(*applet, "description", "applet");
            s.applet.x = extension(*applet, "applet");
        }

        if (auto it = root.find("source"); it != root.end()) {
            if (!it->is_array()) {
                sink_.error("spec.wrong_type", "source must be an array of tables; write each one as [[source]].",
                            "source");
            } else {
                for (std::size_t i = 0; i < it->size(); ++i)
                    s.sources.push_back(source((*it)[i], index_path("source", i)));
            }
        }
        if (auto it = root.find("panel"); it != root.end()) {
            if (!it->is_array()) {
                sink_.error("spec.wrong_type", "panel must be an array of tables; write each one as [[panel]].",
                            "panel");
            } else {
                for (std::size_t i = 0; i < it->size(); ++i)
                    s.panels.push_back(panel((*it)[i], index_path("panel", i)));
            }
        }
        if (auto it = root.find("control"); it != root.end() && version_ >= kInstrumentVersion) {
            if (!it->is_array()) {
                sink_.error("spec.wrong_type",
                            "control must be an array of tables; write each one as [[control]].", "control");
            } else {
                for (std::size_t i = 0; i < it->size(); ++i)
                    s.controls.push_back(control((*it)[i], index_path("control", i)));
            }
        }
        if (version_ >= kInstrumentVersion) {
            if (auto it = root.find("program"); it != root.end()) {
                if (!it->is_array()) {
                    sink_.error("spec.wrong_type",
                                "program must be an array of tables; write each one as [[program]].", "program");
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i)
                        s.programs.push_back(program((*it)[i], index_path("program", i)));
                }
            }
            if (auto it = root.find("stage"); it != root.end()) {
                if (!it->is_array()) {
                    sink_.error("spec.wrong_type", "stage must be an array of tables; write each one as [[stage]].",
                                "stage");
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i)
                        s.stages.push_back(stage((*it)[i], index_path("stage", i)));
                }
            }
            if (auto it = root.find("provenance"); it != root.end()) s.provenance = provenance(*it, "provenance");
        }
        if (auto it = root.find("layout"); it != root.end()) s.layout = layout(*it, "layout");
        s.x = extension(root, "");
        return s;
    }

    // ---- programs, stages and provenance (spec version 1, program.h) ---------
    // The types, the keys, the required keys and the enumerations are checked
    // here; the ranges, the limits and the references are the validator's.

    // Reports every key of `o` that is not in `allowed`, `x` among them: the
    // inner tables of a program carry no extension table.
    void strict_keys(const json& o, const std::string& path, const std::vector<std::string>& allowed,
                     const std::string& owner) {
        for (auto it = o.begin(); it != o.end(); ++it) {
            if (contains(allowed, it.key())) continue;
            sink_.error("spec.unknown_key", quote(it.key()) + " is not a key of " + owner + ".",
                        key_path(path, it.key()));
        }
    }

    // A count of a program: an integer from 0 to 2^32 - 1, which the value
    // type holds; its range is the validator's.
    std::optional<std::uint32_t> count(const json& o, const char* key, const std::string& path) {
        auto v = integer(o, key, path);
        if (!v) return std::nullopt;
        if (*v < 0 || *v > 0xFFFFFFFFll) {
            sink_.error("spec.out_of_range", key_path(path, key) + " must be from 0 to 4294967295.",
                        key_path(path, key));
            return std::nullopt;
        }
        return static_cast<std::uint32_t>(*v);
    }

    std::string required_string(const json& o, const char* key, const std::string& path, const char* owner) {
        if (auto v = string(o, key, path)) return *v;
        if (o.find(key) == o.end()) missing(path, key, owner);
        return std::string();
    }

    Program program(const json& o, const std::string& path) {
        Program p;
        if (!table(o, path, path)) return p;
        keys(o, path, {"id", "title", "wgsl", "entry", "inputs", "output", "limits", "workgroups"}, "a program");
        p.id = required_string(o, "id", path, "A program");
        p.title = string(o, "title", path);
        p.wgsl = required_string(o, "wgsl", path, "A program");
        p.entry = string(o, "entry", path);
        if (auto it = o.find("inputs"); it == o.end()) {
            missing(path, "inputs", "A program");
        } else if (!it->is_array()) {
            wrong(key_path(path, "inputs"), *it, "a list of inputs");
        } else {
            for (std::size_t i = 0; i < it->size(); ++i) {
                const std::string ip = index_path(key_path(path, "inputs"), i);
                const json& item = (*it)[i];
                ProgramInput in;
                if (table(item, ip, ip)) {
                    strict_keys(item, ip, {"var", "source"}, "an input");
                    in.var = required_string(item, "var", ip, "An input");
                    in.source = required_string(item, "source", ip, "An input");
                }
                p.inputs.push_back(std::move(in));
            }
        }
        if (auto it = o.find("output"); it == o.end()) {
            missing(path, "output", "A program");
        } else {
            p.output = output(*it, key_path(path, "output"));
        }
        if (auto it = o.find("limits"); it != o.end()) {
            const std::string lp = key_path(path, "limits");
            if (table(*it, lp, lp)) {
                strict_keys(*it, lp, {"time_ms", "memory_mb", "every"}, "the limits");
                p.limits.time_ms = number(*it, "time_ms", lp);
                p.limits.memory_mb = count(*it, "memory_mb", lp);
                p.limits.every = count(*it, "every", lp);
            }
        }
        if (auto it = o.find("workgroups"); it != o.end()) {
            const std::string wp = key_path(path, "workgroups");
            std::array<std::uint32_t, 3> w{};
            bool ok = it->is_array() && it->size() == 3;
            for (std::size_t i = 0; ok && i < 3; ++i) {
                std::int64_t v = 0;
                ok = as_integer((*it)[i], v) && v >= 0 && v <= 0xFFFFFFFFll;
                if (ok) w[i] = static_cast<std::uint32_t>(v);
            }
            if (ok) {
                p.workgroups = w;
            } else {
                sink_.error("spec.wrong_type", wp + " must be [x, y, z], three workgroup counts.", wp);
            }
        }
        p.x = extension(o, path);
        return p;
    }

    ProgramOutput output(const json& o, const std::string& path) {
        ProgramOutput out;
        if (!table(o, path, path)) return out;
        auto kind = string(o, "kind", path);
        if (!kind) {
            if (o.find("kind") == o.end()) missing(path, "kind", "An output");
            return out;
        }
        auto k = output_kind_from_name(*kind);
        if (!k) {
            sink_.error("spec.bad_value", quote(*kind) + " is not a kind of output; the kinds are geometry and pixels.",
                        key_path(path, "kind"));
            return out;
        }
        out.kind = *k;
        if (out.kind == OutputKind::geometry) {
            strict_keys(o, path, {"kind", "panel", "entity", "part", "max_count", "arrays", "style"},
                        "a geometry output");
            out.panel = required_string(o, "panel", path, "A geometry output");
            out.entity = string(o, "entity", path);
            if (auto part = string(o, "part", path)) {
                if (auto gp = geometry_part_from_name(*part)) {
                    out.part = *gp;
                } else {
                    sink_.error("spec.bad_value",
                                quote(*part) + " is not a part of a geometry output; the parts are points and lines.",
                                key_path(path, "part"));
                }
            }
            if (auto n = count(o, "max_count", path)) {
                out.max_count = *n;
            } else if (o.find("max_count") == o.end()) {
                missing(path, "max_count", "A geometry output");
            }
            if (auto it = o.find("arrays"); it == o.end()) {
                missing(path, "arrays", "A geometry output");
            } else if (!it->is_array()) {
                wrong(key_path(path, "arrays"), *it, "a list of array names");
            } else {
                for (std::size_t i = 0; i < it->size(); ++i) {
                    const json& name = (*it)[i];
                    auto a = name.is_string() ? geometry_array_from_name(name.get<std::string>()) : std::nullopt;
                    if (a) {
                        out.arrays.push_back(*a);
                    } else {
                        sink_.error("spec.bad_value",
                                    "An array of a geometry output is positions, colors, values or radii.",
                                    index_path(key_path(path, "arrays"), i));
                    }
                }
            }
            if (auto it = o.find("style"); it != o.end()) {
                const std::string sp = key_path(path, "style");
                if (table(*it, sp, sp)) {
                    strict_keys(*it, sp, {"colormap", "vmin", "vmax"}, "a style");
                    out.style.colormap = enumerated(*it, "colormap", sp, colormaps(), "a colormap");
                    out.style.vmin = number(*it, "vmin", sp);
                    out.style.vmax = number(*it, "vmax", sp);
                }
            }
        } else {
            strict_keys(o, path, {"kind", "width", "height", "format"}, "a pixels output");
            for (const char* side : {"width", "height"}) {
                auto n = count(o, side, path);
                if (n) {
                    (std::string(side) == "width" ? out.width : out.height) = *n;
                } else if (o.find(side) == o.end()) {
                    missing(path, side, "A pixels output");
                }
            }
            if (auto f = string(o, "format", path)) {
                if (auto pf = pixel_format_from_name(*f)) {
                    out.format = *pf;
                } else {
                    sink_.error("spec.bad_value", quote(*f) + " is not a pixel format; the format is rgba8unorm.",
                                key_path(path, "format"));
                }
            }
        }
        return out;
    }

    Stage stage(const json& o, const std::string& path) {
        Stage st;
        if (!table(o, path, path)) return st;
        keys(o, path, {"panel", "entity", "kind", "wgsl"}, "a stage");
        st.panel = required_string(o, "panel", path, "A stage");
        st.entity = required_string(o, "entity", path, "A stage");
        if (auto kind = string(o, "kind", path)) {
            if (auto k = stage_kind_from_name(*kind)) {
                st.kind = *k;
            } else {
                sink_.error("spec.bad_value", quote(*kind) + " is not a kind of stage; the kinds are material and vertex.",
                            key_path(path, "kind"));
            }
        } else if (o.find("kind") == o.end()) {
            missing(path, "kind", "A stage");
        }
        st.wgsl = required_string(o, "wgsl", path, "A stage");
        st.x = extension(o, path);
        return st;
    }

    Provenance provenance(const json& o, const std::string& path) {
        Provenance pv;
        if (!table(o, path, path)) return pv;
        keys(o, path, {"foreign", "from", "reviewed"}, "the provenance table");
        if (auto f = boolean(o, "foreign", path)) {
            pv.foreign = *f;
        } else if (o.find("foreign") == o.end()) {
            missing(path, "foreign", "The provenance table");
        }
        pv.from = string(o, "from", path);
        if (auto it = o.find("reviewed"); it != o.end()) {
            const std::string rp = key_path(path, "reviewed");
            if (!it->is_array()) {
                wrong(rp, *it, "a list of reviewed sources");
            } else {
                for (std::size_t i = 0; i < it->size(); ++i) {
                    const std::string ip = index_path(rp, i);
                    const json& item = (*it)[i];
                    ReviewedSource r;
                    if (table(item, ip, ip)) {
                        strict_keys(item, ip, {"target", "sha256"}, "a reviewed source");
                        r.target = required_string(item, "target", ip, "A reviewed source");
                        r.sha256 = required_string(item, "sha256", ip, "A reviewed source");
                    }
                    pv.reviewed.push_back(std::move(r));
                }
            }
        }
        pv.x = extension(o, path);
        return pv;
    }

    // One entry of the control array (spec version 1, instrument.h). The kind
    // is checked here; the target grammar, the kind's targets and options and
    // the ranges are the validator's.
    Control control(const json& o, const std::string& path) {
        Control c;
        if (!table(o, path, path)) return c;
        keys(o, path, {"id", "kind", "title", "bind", "options"}, "a control");
        if (auto id = string(o, "id", path)) {
            c.id = *id;
        } else if (o.find("id") == o.end()) {
            missing(path, "id", "A control");
        }
        if (auto kind = string(o, "kind", path)) {
            if (auto k = control_kind_from_name(*kind)) {
                c.kind = *k;
            } else {
                sink_.error("spec.unknown_control_kind",
                            quote(*kind) + " is not a kind of control; the kinds are " + join(control_kind_names()) +
                                ".",
                            key_path(path, "kind"));
            }
        } else if (o.find("kind") == o.end()) {
            missing(path, "kind", "A control");
        }
        c.title = string(o, "title", path);
        if (auto bind = string(o, "bind", path)) {
            c.bind = *bind;
        } else if (o.find("bind") == o.end()) {
            missing(path, "bind", "A control");
        }
        if (auto it = o.find("options"); it != o.end()) {
            const std::string op = key_path(path, "options");
            if (table(*it, op, op)) {
                const json& opts = *it;
                for (auto k = opts.begin(); k != opts.end(); ++k) {
                    if (k.key() == "x" || contains(control_option_names(), k.key())) continue;
                    sink_.error("spec.unknown_option",
                                quote(k.key()) + " is not an option of a control; its options are " +
                                    join(control_option_names()) + ".",
                                key_path(op, k.key()));
                }
                c.options.min = number(opts, "min", op);
                c.options.max = number(opts, "max", op);
                c.options.step = number(opts, "step", op);
                c.options.log = boolean(opts, "log", op);
                c.options.value = number(opts, "value", op);
                c.options.x = extension(opts, op);
            }
        }
        c.x = extension(o, path);
        return c;
    }

    // A list of 1 to 64 entity path patterns; the count and the lengths are
    // the validator's.
    std::vector<std::string> patterns(const json& opts, const char* key, const std::string& op) {
        std::vector<std::string> out;
        const json& list = opts[key];
        const std::string lp = key_path(op, key);
        if (!list.is_array()) {
            wrong(lp, list, "a list of entity path patterns");
            return out;
        }
        for (std::size_t i = 0; i < list.size(); ++i) {
            if (list[i].is_string()) {
                out.push_back(list[i].get<std::string>());
            } else {
                wrong(index_path(lp, i), list[i], "an entity path pattern");
            }
        }
        if (list.empty())
            sink_.error("spec.bad_value", lp + " lists no pattern; leave it out to draw every entity.", lp);
        return out;
    }

    // An orbit table {target, distance, azimuth, elevation} or a camera's
    // entity path, kept as written; the ranges are the validator's.
    json view_camera(const json& v, const std::string& path) {
        if (v.is_string()) return v;
        if (!v.is_object()) {
            wrong(path, v, "an orbit table or a camera's entity path");
            return json();
        }
        keys(v, path, {"target", "distance", "azimuth", "elevation"}, "an orbit table");
        json out = json::object();
        if (auto t = v.find("target"); t != v.end()) {
            if (!t->is_array() || t->size() != 3 || !(*t)[0].is_number() || !(*t)[1].is_number() ||
                !(*t)[2].is_number()) {
                sink_.error("spec.wrong_type", key_path(path, "target") + " must be [x, y, z], three numbers.",
                            key_path(path, "target"));
            } else {
                out["target"] = *t;
            }
        }
        for (const char* k : {"distance", "azimuth", "elevation"})
            if (auto n = number(v, k, path)) out[k] = *n;
        if (v.find("x") != v.end())
            sink_.error("spec.unknown_key", "\"x\" is not a key of an orbit table.", key_path(path, "x"));
        return out;
    }

    Source source(const json& o, const std::string& path) {
        Source s;
        if (!table(o, path, path)) return s;
        keys(o, path, {"id", "catalog", "sql", "from", "transform"}, "a source");
        if (auto id = string(o, "id", path)) {
            s.id = *id;
        } else if (o.find("id") == o.end()) {
            missing(path, "id", "A source");
        }
        s.catalog = string(o, "catalog", path);
        s.sql = string(o, "sql", path);
        s.from = string(o, "from", path);
        if (auto it = o.find("transform"); it != o.end()) {
            const std::string tp = key_path(path, "transform");
            if (!it->is_array()) {
                sink_.error("spec.wrong_type",
                            tp + " must be an array of tables; write each one as [[source.transform]].", tp);
            } else {
                for (std::size_t i = 0; i < it->size(); ++i)
                    s.transforms.push_back(transform((*it)[i], index_path(tp, i)));
            }
        }
        s.x = extension(o, path);
        return s;
    }

    Transform transform(const json& o, const std::string& path) {
        Transform t;
        if (!table(o, path, path)) return t;
        auto op = string(o, "op", path);
        if (!op) {
            if (o.find("op") == o.end()) missing(path, "op", "A transform");
            return t;
        }
        if (!transform_op_from_string(*op, t.op)) {
            sink_.error("spec.unknown_transform",
                        quote(*op) + " is not a transform; the transforms are slice, reduce and normalise.",
                        key_path(path, "op"));
            return t;
        }
        std::vector<std::string> allowed = transform_keys(t.op);
        allowed.emplace_back("op");
        const std::string owner = std::string("a ") + to_string(t.op) + " transform";
        keys(o, path, allowed, owner);

        if (auto it = o.find("axis"); it != o.end()) {
            const std::string ap = key_path(path, "axis");
            std::int64_t a = 0;
            if (as_integer(*it, a)) {
                t.axis.push_back(a);
            } else if (it->is_array() && t.op == TransformOp::reduce) {
                t.axis_list = true;
                for (std::size_t i = 0; i < it->size(); ++i) {
                    if (as_integer((*it)[i], a)) {
                        t.axis.push_back(a);
                    } else {
                        wrong(index_path(ap, i), (*it)[i], "an integer");
                    }
                }
            } else {
                wrong(ap, *it, t.op == TransformOp::reduce ? "an integer or a list of integers" : "an integer");
            }
        } else if (t.op != TransformOp::normalise) {
            missing(path, "axis", std::string("A ") + to_string(t.op) + " transform");
        }
        if (t.op == TransformOp::slice) {
            t.index = integer(o, "index", path);
            t.start = integer(o, "start", path);
            t.stop = integer(o, "stop", path);
            t.step = integer(o, "step", path);
        }
        if (t.op == TransformOp::reduce) {
            t.fn = enumerated(o, "fn", path, reduce_fns(), "a reduction");
            if (o.find("fn") == o.end()) missing(path, "fn", "A reduce transform");
            t.keepdims = boolean(o, "keepdims", path);
        }
        if (t.op == TransformOp::normalise) {
            t.mode = enumerated(o, "mode", path, normalise_modes(), "a normalise mode");
            if (o.find("mode") == o.end()) missing(path, "mode", "A normalise transform");
        }
        t.x = extension(o, path);
        return t;
    }

    Panel panel(const json& o, const std::string& path) {
        Panel p;
        if (!table(o, path, path)) return p;
        keys(o, path, {"id", "primitive", "title", "bind", "options"}, "a panel");
        if (auto id = string(o, "id", path)) {
            p.id = *id;
        } else if (o.find("id") == o.end()) {
            missing(path, "id", "A panel");
        }
        bool known = false;
        if (auto prim = string(o, "primitive", path)) {
            known = primitive_from_string(*prim, p.primitive) && primitive_version(p.primitive) <= version_;
            if (!known)
                sink_.error("spec.unknown_primitive",
                            quote(*prim) + " is not a primitive" +
                                (primitive_from_string(*prim, p.primitive) ? " of format version " +
                                                                                 std::to_string(version_)
                                                                           : std::string()) +
                                "; the primitives are " + join(primitive_names(version_)) + ".",
                            key_path(path, "primitive"));
        } else if (o.find("primitive") == o.end()) {
            missing(path, "primitive", "A panel");
        }
        p.title = string(o, "title", path);
        p.x = extension(o, path);
        if (!known) return p;  // slots and options depend on the primitive
        const std::string prim_name = to_string(p.primitive);

        if (auto it = o.find("bind"); it != o.end()) {
            const std::string bp = key_path(path, "bind");
            if (table(*it, bp, bp)) {
                for (auto s = it->begin(); s != it->end(); ++s) {
                    const std::string sp = key_path(bp, s.key());
                    if (!contains(slots_of(p.primitive), s.key())) {
                        sink_.error("spec.unknown_slot",
                                    quote(s.key()) + " is not a slot of a " + prim_name + "; its slots are " +
                                        join(slots_of(p.primitive)) + ".",
                                    sp);
                        continue;
                    }
                    SlotValue value;
                    if (s->is_string()) {
                        value.bindings.push_back(s->get<std::string>());
                    } else if (s->is_array()) {
                        value.list = true;
                        for (std::size_t i = 0; i < s->size(); ++i) {
                            if ((*s)[i].is_string()) {
                                value.bindings.push_back((*s)[i].get<std::string>());
                            } else {
                                wrong(index_path(sp, i), (*s)[i], "a binding string");
                            }
                        }
                    } else {
                        wrong(sp, *s, "a binding string or a list of them");
                        continue;
                    }
                    p.bind[s.key()] = std::move(value);
                }
            }
        }

        if (auto it = o.find("options"); it != o.end()) {
            const std::string op = key_path(path, "options");
            if (table(*it, op, op)) {
                const json& opts = *it;
                const auto& allowed = options_of(p.primitive);
                for (auto k = opts.begin(); k != opts.end(); ++k) {
                    if (k.key() == "x" || contains(allowed, k.key())) continue;
                    sink_.error("spec.unknown_option",
                                quote(k.key()) + " is not an option of a " + prim_name + "; its options are " +
                                    join(allowed) + ".",
                                key_path(op, k.key()));
                }
                Options& out = p.options;
                auto has = [&](const char* k) { return contains(allowed, k) && opts.find(k) != opts.end(); };
                if (has("x_label")) out.x_label = string(opts, "x_label", op);
                if (has("y_label")) out.y_label = string(opts, "y_label", op);
                if (has("z_label")) out.z_label = string(opts, "z_label", op);
                if (has("x_range")) out.x_range = range(opts, "x_range", op);
                if (has("y_range")) out.y_range = range(opts, "y_range", op);
                if (has("range")) out.range = range(opts, "range", op);
                if (has("x_log")) out.x_log = boolean(opts, "x_log", op);
                if (has("y_log")) out.y_log = boolean(opts, "y_log", op);
                if (has("colormap")) out.colormap = enumerated(opts, "colormap", op, colormaps(), "a colormap");
                if (has("bins")) out.bins = integer(opts, "bins", op);
                if (has("text")) out.text = string(opts, "text", op);
                if (has("up")) out.scene.up = enumerated(opts, "up", op, scene_up_axes(), "an axis");
                if (has("view_camera")) out.scene.view_camera = view_camera(opts["view_camera"], key_path(op, "view_camera"));
                if (has("include")) out.scene.include = patterns(opts, "include", op);
                if (has("exclude")) out.scene.exclude = patterns(opts, "exclude", op);
                if (has("labels")) out.scene.labels = boolean(opts, "labels", op);
                if (has("axes")) out.scene.axes = boolean(opts, "axes", op);
                if (has("scale_bar")) out.scene.scale_bar = boolean(opts, "scale_bar", op);
                if (has("columns")) {
                    const json& c = opts["columns"];
                    const std::string cp = key_path(op, "columns");
                    if (!c.is_array()) {
                        wrong(cp, c, "a list of column names");
                    } else {
                        std::vector<std::string> names;
                        for (std::size_t i = 0; i < c.size(); ++i) {
                            if (c[i].is_string()) {
                                names.push_back(c[i].get<std::string>());
                            } else {
                                wrong(index_path(cp, i), c[i], "a column name");
                            }
                        }
                        out.columns = std::move(names);
                    }
                }
                out.x = extension(opts, op);
            }
        }
        return p;
    }

    LayoutNode layout(const json& v, const std::string& path) {
        LayoutNode n;
        if (v.is_string()) {
            n.kind = LayoutNode::Kind::panel;
            n.panel = v.get<std::string>();
            return n;
        }
        if (!v.is_object()) {
            sink_.error("spec.wrong_type",
                        path + " must be a panel id or a table with split or tabs, not " + type_name(v) + ".", path);
            return n;
        }
        const bool has_split = v.find("split") != v.end();
        const bool has_tabs = v.find("tabs") != v.end();
        if (has_split == has_tabs) {
            sink_.error("spec.bad_layout",
                        has_split ? path + " has both split and tabs; a layout table has one of them."
                                  : path + " needs split, with children, or tabs.",
                        path);
            return n;
        }
        n.title = string(v, "title", path);
        n.x = extension(v, path);
        if (has_split) {
            n.kind = LayoutNode::Kind::split;
            keys(v, path, {"split", "ratio", "title", "children"}, "a split");
            static const std::vector<std::string> directions = {"horizontal", "vertical"};
            if (auto d = enumerated(v, "split", path, directions, "a split direction"))
                split_from_string(*d, n.split);
            n.ratio = number(v, "ratio", path);
            auto it = v.find("children");
            if (it == v.end()) {
                missing(path, "children", "A split");
            } else {
                children(*it, key_path(path, "children"), n);
            }
        } else {
            n.kind = LayoutNode::Kind::tabs;
            keys(v, path, {"tabs", "title"}, "tabs");
            children(v["tabs"], key_path(path, "tabs"), n);
        }
        return n;
    }

private:
    void children(const json& list, const std::string& path, LayoutNode& n) {
        if (!list.is_array()) {
            wrong(path, list, "a list of layout nodes");
            return;
        }
        for (std::size_t i = 0; i < list.size(); ++i) n.children.push_back(layout(list[i], index_path(path, i)));
    }

    Sink& sink_;
    std::int64_t version_ = kLatestFormatVersion;
};

}  // namespace

Source decode_source(const json& item, const std::string& path, Sink& sink) {
    return Decoder(sink).source(item, path);
}

Panel decode_panel(const json& item, const std::string& path, Sink& sink) { return Decoder(sink).panel(item, path); }

LayoutNode decode_layout(const json& item, const std::string& path, Sink& sink) {
    return Decoder(sink).layout(item, path);
}

Control decode_control(const json& item, const std::string& path, Sink& sink) {
    return Decoder(sink).control(item, path);
}

Program decode_program(const json& item, const std::string& path, Sink& sink) {
    return Decoder(sink).program(item, path);
}

Stage decode_stage(const json& item, const std::string& path, Sink& sink) { return Decoder(sink).stage(item, path); }

}  // namespace detail

using detail::json;

namespace {

// Thrown inside the JSON parser's callback to stop a document that nests too
// deeply, or that repeats a key in one object; caught in from_json_text.
struct TooDeep {};
struct DuplicateKey {
    std::string key;
};

Position position_of_byte(std::string_view text, std::size_t byte) {
    Position p{1, 1};
    const std::size_t end = byte < text.size() ? byte : text.size();
    for (std::size_t i = 0; i < end; ++i) {
        if (text[i] == '\n') {
            ++p.line;
            p.column = 1;
        } else if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
            ++p.column;
        }
    }
    return p;
}

// `depth` is the item's depth in a whole document: 2 for a source or a panel
// (the root table, then the array), 1 for the layout.
template <class T, class F>
Decoded<T> decode_item(const json& item, const std::string& path, int depth, F decode) {
    Decoded<T> out;
    try {
        detail::Sink sink(out.diagnostics);
        std::size_t count = 0;
        if (!detail::check_values(item, path, depth, count, sink)) return out;
        T value = decode(item, path, sink);
        if (sink.errors() == 0) out.value = std::move(value);
    } catch (...) {
        out.diagnostics.push_back({Severity::error, "spec.internal", "The spec library failed unexpectedly.", path});
    }
    return out;
}

}  // namespace

ParseResult from_json(const json& document) {
    ParseResult r;
    try {
        detail::Sink sink(r.diagnostics);
        std::size_t count = 0;
        if (!detail::check_values(document, "", 0, count, sink)) return r;
        Spec spec = detail::Decoder(sink).spec(document);
        if (sink.errors() == 0) r.spec = std::move(spec);
    } catch (...) {
        r.diagnostics.push_back({Severity::error, "spec.internal", "The spec library failed unexpectedly.", ""});
    }
    return r;
}

ParseResult from_json_text(std::string_view text) {
    ParseResult r;
    if (text.size() > limits::kTextBytes) {
        r.diagnostics.push_back({Severity::error, "spec.too_large",
                                 "The spec is " + std::to_string(text.size()) + " bytes long; the limit is " +
                                     std::to_string(limits::kTextBytes) + " bytes.",
                                 ""});
        return r;
    }
    json doc;
    try {
        // Track the keys of each open object to refuse duplicates, which the
        // JSON parser would otherwise resolve silently, and stop deep nesting.
        std::vector<std::set<std::string>> open;
        auto callback = [&open](int depth, json::parse_event_t event, json& parsed) {
            switch (event) {
                case json::parse_event_t::object_start:
                    if (depth > limits::kDepth + 1) throw TooDeep{};
                    open.emplace_back();
                    break;
                case json::parse_event_t::array_start:
                    if (depth > limits::kDepth + 1) throw TooDeep{};
                    break;
                case json::parse_event_t::object_end:
                    if (!open.empty()) open.pop_back();
                    break;
                case json::parse_event_t::key:
                    if (!open.empty() && !open.back().insert(parsed.get<std::string>()).second)
                        throw DuplicateKey{parsed.get<std::string>()};
                    break;
                default: break;
            }
            return true;
        };
        doc = json::parse(text.begin(), text.end(), callback);
    } catch (const TooDeep&) {
        r.diagnostics.push_back({Severity::error, "spec.too_deep",
                                 "The document nests more than " + std::to_string(limits::kDepth) +
                                     " levels of tables and arrays.",
                                 ""});
        return r;
    } catch (const json::parse_error& e) {
        Diagnostic d{Severity::error, "spec.syntax", "The text is not valid JSON.", ""};
        const Position p = position_of_byte(text, e.byte > 0 ? e.byte - 1 : 0);
        d.line = p.line;
        d.column = p.column;
        r.diagnostics.push_back(d);
        return r;
    } catch (const DuplicateKey& e) {
        r.diagnostics.push_back({Severity::error, "spec.syntax",
                                 "The JSON text repeats the key " + detail::quote(e.key) + " in one object.", ""});
        return r;
    } catch (...) {
        r.diagnostics.push_back({Severity::error, "spec.syntax", "The text is not valid JSON.", ""});
        return r;
    }
    return from_json(doc);
}

Decoded<Source> source_from_json(const json& item, const std::string& path) {
    return decode_item<Source>(item, path, 2, detail::decode_source);
}

Decoded<Panel> panel_from_json(const json& item, const std::string& path) {
    return decode_item<Panel>(item, path, 2, detail::decode_panel);
}

Decoded<LayoutNode> layout_from_json(const json& item, const std::string& path) {
    return decode_item<LayoutNode>(item, path, 1, detail::decode_layout);
}

Decoded<Control> control_from_json(const json& item, const std::string& path) {
    return decode_item<Control>(item, path, 2, detail::decode_control);
}

Decoded<Program> program_from_json(const json& item, const std::string& path) {
    return decode_item<Program>(item, path, 2, detail::decode_program);
}

Decoded<Stage> stage_from_json(const json& item, const std::string& path) {
    return decode_item<Stage>(item, path, 2, detail::decode_stage);
}

// ---- writing ----------------------------------------------------------------

namespace {

json transform_json(const Transform& t) {
    json j = json::object();
    j["op"] = to_string(t.op);
    if (!t.axis.empty()) {
        if (t.axis_list || t.axis.size() != 1) {
            j["axis"] = t.axis;
        } else {
            j["axis"] = t.axis[0];
        }
    }
    if (t.index) j["index"] = *t.index;
    if (t.start) j["start"] = *t.start;
    if (t.stop) j["stop"] = *t.stop;
    if (t.step) j["step"] = *t.step;
    if (t.fn) j["fn"] = *t.fn;
    if (t.keepdims) j["keepdims"] = *t.keepdims;
    if (t.mode) j["mode"] = *t.mode;
    if (!t.x.is_null()) j["x"] = t.x;
    return j;
}

json options_json(const Options& o) {
    json j = json::object();
    auto pair = [](const std::array<double, 2>& r) { return json::array({r[0], r[1]}); };
    if (o.x_label) j["x_label"] = *o.x_label;
    if (o.y_label) j["y_label"] = *o.y_label;
    if (o.z_label) j["z_label"] = *o.z_label;
    if (o.x_range) j["x_range"] = pair(*o.x_range);
    if (o.y_range) j["y_range"] = pair(*o.y_range);
    if (o.range) j["range"] = pair(*o.range);
    if (o.x_log) j["x_log"] = *o.x_log;
    if (o.y_log) j["y_log"] = *o.y_log;
    if (o.colormap) j["colormap"] = *o.colormap;
    if (o.bins) j["bins"] = *o.bins;
    if (o.columns) j["columns"] = *o.columns;
    if (o.text) j["text"] = *o.text;
    const SceneOptions& sc = o.scene;
    if (sc.up) j["up"] = *sc.up;
    if (!sc.view_camera.is_null()) j["view_camera"] = sc.view_camera;
    if (!sc.include.empty()) j["include"] = sc.include;
    if (!sc.exclude.empty()) j["exclude"] = sc.exclude;
    if (sc.labels) j["labels"] = *sc.labels;
    if (sc.axes) j["axes"] = *sc.axes;
    if (sc.scale_bar) j["scale_bar"] = *sc.scale_bar;
    if (!o.x.is_null()) j["x"] = o.x;
    return j;
}

json control_options_json(const ControlOptions& o) {
    json j = json::object();
    if (o.min) j["min"] = *o.min;
    if (o.max) j["max"] = *o.max;
    if (o.step) j["step"] = *o.step;
    if (o.log) j["log"] = *o.log;
    if (o.value) j["value"] = *o.value;
    if (!o.x.is_null()) j["x"] = o.x;
    return j;
}

}  // namespace

json to_json(const Source& s) {
    json j = json::object();
    j["id"] = s.id;
    if (s.catalog) j["catalog"] = *s.catalog;
    if (s.sql) j["sql"] = *s.sql;
    if (s.from) j["from"] = *s.from;
    if (!s.transforms.empty()) {
        json list = json::array();
        for (const Transform& t : s.transforms) list.push_back(transform_json(t));
        j["transform"] = std::move(list);
    }
    if (!s.x.is_null()) j["x"] = s.x;
    return j;
}

json to_json(const Panel& p) {
    json j = json::object();
    j["id"] = p.id;
    j["primitive"] = to_string(p.primitive);
    if (p.title) j["title"] = *p.title;
    if (!p.bind.empty()) {
        json bind = json::object();
        for (const auto& [slot, value] : p.bind) {
            if (!value.list && value.bindings.size() == 1) {
                bind[slot] = value.bindings[0];
            } else {
                bind[slot] = value.bindings;
            }
        }
        j["bind"] = std::move(bind);
    }
    if (!p.options.empty()) j["options"] = options_json(p.options);
    if (!p.x.is_null()) j["x"] = p.x;
    return j;
}

json to_json(const Control& c) {
    json j = json::object();
    j["id"] = c.id;
    j["kind"] = control_kind_name(c.kind);
    if (c.title) j["title"] = *c.title;
    j["bind"] = c.bind;
    if (!c.options.empty()) j["options"] = control_options_json(c.options);
    if (!c.x.is_null()) j["x"] = c.x;
    return j;
}

json to_json(const Program& p) {
    json j = json::object();
    j["id"] = p.id;
    if (p.title) j["title"] = *p.title;
    j["wgsl"] = p.wgsl;
    if (p.entry) j["entry"] = *p.entry;
    json inputs = json::array();
    for (const ProgramInput& in : p.inputs) inputs.push_back(json{{"var", in.var}, {"source", in.source}});
    j["inputs"] = std::move(inputs);
    const ProgramOutput& o = p.output;
    json out = json::object();
    out["kind"] = output_kind_name(o.kind);
    if (o.kind == OutputKind::geometry) {
        out["panel"] = o.panel;
        if (o.entity) out["entity"] = *o.entity;
        out["part"] = geometry_part_name(o.part);
        out["max_count"] = o.max_count;
        json arrays = json::array();
        for (GeometryArray a : o.arrays) arrays.push_back(geometry_array_name(a));
        out["arrays"] = std::move(arrays);
        json style = json::object();
        if (o.style.colormap) style["colormap"] = *o.style.colormap;
        if (o.style.vmin) style["vmin"] = *o.style.vmin;
        if (o.style.vmax) style["vmax"] = *o.style.vmax;
        if (!style.empty()) out["style"] = std::move(style);
    } else {
        out["width"] = o.width;
        out["height"] = o.height;
        out["format"] = pixel_format_name(o.format);
    }
    j["output"] = std::move(out);
    json limits = json::object();
    if (p.limits.time_ms) limits["time_ms"] = *p.limits.time_ms;
    if (p.limits.memory_mb) limits["memory_mb"] = *p.limits.memory_mb;
    if (p.limits.every) limits["every"] = *p.limits.every;
    if (!limits.empty()) j["limits"] = std::move(limits);
    if (p.workgroups) j["workgroups"] = json::array({(*p.workgroups)[0], (*p.workgroups)[1], (*p.workgroups)[2]});
    if (!p.x.is_null()) j["x"] = p.x;
    return j;
}

json to_json(const Stage& s) {
    json j = json::object();
    j["panel"] = s.panel;
    j["entity"] = s.entity;
    j["kind"] = stage_kind_name(s.kind);
    j["wgsl"] = s.wgsl;
    if (!s.x.is_null()) j["x"] = s.x;
    return j;
}

json to_json(const Provenance& p) {
    json j = json::object();
    j["foreign"] = p.foreign;
    if (p.from) j["from"] = *p.from;
    if (!p.reviewed.empty()) {
        json list = json::array();
        for (const ReviewedSource& r : p.reviewed) list.push_back(json{{"target", r.target}, {"sha256", r.sha256}});
        j["reviewed"] = std::move(list);
    }
    if (!p.x.is_null()) j["x"] = p.x;
    return j;
}

json to_json(const LayoutNode& n) {
    if (n.kind == LayoutNode::Kind::panel) return n.panel;
    json j = json::object();
    json children = json::array();
    for (const LayoutNode& c : n.children) children.push_back(to_json(c));
    if (n.kind == LayoutNode::Kind::split) {
        j["split"] = to_string(n.split);
        if (n.ratio) j["ratio"] = *n.ratio;
        j["children"] = std::move(children);
    } else {
        j["tabs"] = std::move(children);
    }
    if (n.title) j["title"] = *n.title;
    if (!n.x.is_null()) j["x"] = n.x;
    return j;
}

json to_json(const Spec& spec) {
    json j = json::object();
    j["caliper_spec"] = spec.version;
    json applet = json::object();
    applet["id"] = spec.applet.id;
    if (spec.applet.title) applet["title"] = *spec.applet.title;
    if (spec.applet.description) applet["description"] = *spec.applet.description;
    if (!spec.applet.x.is_null()) applet["x"] = spec.applet.x;
    j["applet"] = std::move(applet);
    if (!spec.sources.empty()) {
        json list = json::array();
        for (const Source& s : spec.sources) list.push_back(to_json(s));
        j["source"] = std::move(list);
    }
    if (!spec.panels.empty()) {
        json list = json::array();
        for (const Panel& p : spec.panels) list.push_back(to_json(p));
        j["panel"] = std::move(list);
    }
    if (!spec.controls.empty()) {
        json list = json::array();
        for (const Control& c : spec.controls) list.push_back(to_json(c));
        j["control"] = std::move(list);
    }
    if (!spec.programs.empty()) {
        json list = json::array();
        for (const Program& p : spec.programs) list.push_back(to_json(p));
        j["program"] = std::move(list);
    }
    if (!spec.stages.empty()) {
        json list = json::array();
        for (const Stage& s : spec.stages) list.push_back(to_json(s));
        j["stage"] = std::move(list);
    }
    if (!spec.provenance.empty()) j["provenance"] = to_json(spec.provenance);
    if (spec.layout) j["layout"] = to_json(*spec.layout);
    if (!spec.x.is_null()) j["x"] = spec.x;
    return j;
}

std::string to_json_text(const Spec& spec) {
    try {
        return to_json(spec).dump(-1, ' ', false, json::error_handler_t::replace);
    } catch (...) {
        return std::string();
    }
}

}  // namespace caliper_builder::spec
