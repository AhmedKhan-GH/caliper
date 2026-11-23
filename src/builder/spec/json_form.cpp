// The canonical JSON form of a spec: decoding a JSON tree into a Spec (the
// one decoder, which parse_toml also uses after it turns TOML into JSON) and
// writing a Spec back as JSON.
#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/format.h>

#include <algorithm>
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

// Whether `name` is an option that format version 1 adds to `p`: a document of
// version 0 reads it as an unknown option.
bool version_one_option(Primitive p, const std::string& name) {
    for (const char* n : {"legend", "box", "views", "select", "format", "fields"})
        if (name == n) return true;
    return name == "scale_bar" && p != Primitive::scene;
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

    // A binding reference at a key that takes a literal is
    // spec.reference_not_allowed (design/instrument-interface.md, Reference
    // Places) from version 1 on; any other value of the wrong type is
    // spec.wrong_type, as is a reference in a version 0 document.
    void wrong(const std::string& path, const json& v, const std::string& expected) {
        if (version_ >= kInstrumentVersion && looks_like_reference(v)) {
            sink_.error("spec.reference_not_allowed",
                        path + " takes " + expected +
                            " only; a reference stands only where a value changes data, never where it changes the "
                            "shape of the spec.",
                        path);
            return;
        }
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

    // ---- binding references (spec version 1, reference.h) -----------------------

    enum class Ref { none, read, bad };

    // Reads the table `v` at `path` as a reference when the version allows one
    // and the table has the form of one. A table of that form that is not a
    // reference that parse_reference() reads, or that holds a key outside the
    // form or a nested reference, is spec.bad_reference.
    Ref reference(const json& v, const std::string& path, Reference& out) {
        if (version_ < kInstrumentVersion || !looks_like_reference(v)) return Ref::none;
        bool ok = false;
        if (auto r = parse_reference(v)) {
            ok = !looks_like_reference(r->fallback);
            for (auto it = v.begin(); ok && it != v.end(); ++it) {
                bool known = false;
                for (const char* k : kReferenceInputKeys) known = known || it.key() == k;
                for (const char* k : kReferenceOtherKeys) known = known || it.key() == k;
                ok = known;
            }
            if (ok) out = std::move(*r);
        }
        if (!ok) {
            sink_.error("spec.bad_reference",
                        (path.empty() ? std::string("A reference") : path) +
                            " is not a binding reference; it needs one of control, selection and state with a text "
                            "value, and may add field, entity and default.",
                        path);
            return Ref::bad;
        }
        return Ref::read;
    }

    // The value of `key` read by `literal`, or a reference that goes to `refs`
    // in its place, which leaves the literal empty.
    template <class F>
    auto or_reference(const json& o, const char* key, const std::string& path, std::map<std::string, Reference>& refs,
                      F literal) -> decltype(literal()) {
        using R = decltype(literal());
        auto it = o.find(key);
        if (it == o.end()) return R{};
        Reference r;
        switch (reference(*it, key_path(path, key), r)) {
            case Ref::read: refs[key] = std::move(r); return R{};
            case Ref::bad: return R{};
            case Ref::none: break;
        }
        return literal();
    }

    // The keys `<format_key>` and `fields` of `o` as one bound text, or empty
    // when neither is there. A field is a source id or a reference.
    std::optional<BoundText> bound_text(const json& o, const char* format_key, const std::string& path,
                                        const std::string& owner) {
        const bool has_format = o.find(format_key) != o.end();
        const bool has_fields = o.find("fields") != o.end();
        if (!has_format && !has_fields) return std::nullopt;
        BoundText t;
        bool ok = true;
        if (has_format) {
            if (auto f = string(o, format_key, path)) {
                t.format = *f;
            } else {
                ok = false;
            }
        } else {
            missing(path, format_key, owner);
            ok = false;
        }
        if (has_fields) {
            const json& fields = o.at("fields");
            const std::string fp = key_path(path, "fields");
            if (!fields.is_object()) {
                wrong(fp, fields, "a table of fields");
                ok = false;
            } else {
                for (auto it = fields.begin(); it != fields.end(); ++it) {
                    const std::string ep = key_path(fp, it.key());
                    Reference r;
                    if (it->is_string()) {
                        t.fields[it.key()] = it->get<std::string>();
                    } else {
                        switch (reference(*it, ep, r)) {
                            case Ref::read: t.fields[it.key()] = std::move(r); break;
                            case Ref::bad: ok = false; break;
                            case Ref::none:
                                wrong(ep, *it, "a source id or a reference");
                                ok = false;
                                break;
                        }
                    }
                }
            }
        }
        if (!ok) return std::nullopt;
        return t;
    }

    // The names and numbers of a `params` table: a number or a reference each.
    Params params(const json& o, const std::string& path) {
        Params out;
        auto it = o.find("params");
        if (it == o.end()) return out;
        const std::string pp = key_path(path, "params");
        if (!table(*it, pp, pp)) return out;
        for (auto e = it->begin(); e != it->end(); ++e) {
            const std::string ep = key_path(pp, e.key());
            Reference r;
            if (e->is_number()) {
                out[e.key()] = e->get<double>();
            } else {
                switch (reference(*e, ep, r)) {
                    case Ref::read: out[e.key()] = std::move(r); break;
                    case Ref::bad: break;
                    case Ref::none: wrong(ep, *e, "a number or a reference"); break;
                }
            }
        }
        return out;
    }

    // A list of numbers of fixed length: a pose's vectors and a view's target.
    template <std::size_t N>
    void numbers(const json& o, const char* key, const std::string& path, std::array<double, N>& out) {
        auto it = o.find(key);
        if (it == o.end()) return;
        const std::string kp = key_path(path, key);
        bool ok = it->is_array() && it->size() == N;
        for (std::size_t i = 0; ok && i < N; ++i) ok = (*it)[i].is_number();
        if (!ok) {
            sink_.error("spec.wrong_type", kp + " must be a list of " + std::to_string(N) + " numbers.", kp);
            return;
        }
        for (std::size_t i = 0; i < N; ++i) out[i] = (*it)[i].get<double>();
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
                 {"caliper_spec", "applet", "source", "panel", "control", "state", "program", "stage", "provenance",
                  "layout", kPiecesKey},
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
            if (auto it = root.find("state"); it != root.end()) {
                if (!it->is_array()) {
                    sink_.error("spec.wrong_type", "state must be an array of tables; write each one as [[state]].",
                                "state");
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i)
                        s.states.push_back(state((*it)[i], index_path("state", i)));
                }
            }
            if (auto it = root.find("program"); it != root.end()) {
                if (!it->is_array()) {
                    sink_.error("spec.wrong_type",
                                "program must be an array of tables; write each one as [[program]].", "program");
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i)
                        s.programs.push_back(program((*it)[i], index_path("program", i)));
                }
            }
            // Inline pieces (design/programmable-graphics.md, Piece Files).
            if (auto it = root.find(kPiecesKey); it != root.end()) {
                if (!it->is_array()) {
                    sink_.error("spec.wrong_type", "piece must be an array of tables; write each one as [[piece]].",
                                kPiecesKey);
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i)
                        s.pieces.push_back(piece((*it)[i], index_path(kPiecesKey, i)));
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
        // An instance of a piece (design/programmable-graphics.md, Program
        // Instances) holds a pin and none of the keys of the one-pass form.
        bool instance = false;
        for (const char* k : kInstanceKeys) instance = instance || o.find(k) != o.end();
        if (instance) return instance_program(o, path);
        keys(o, path, {"id", "title", "wgsl", "entry", "inputs", "output", "limits", "workgroups", "params"},
             "a program");
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
        p.params = params(o, path);
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
            sink_.error("spec.bad_value", quote(*kind) + " is not a kind of output; the kinds are geometry, pixels and tensor.",
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
        } else if (out.kind == OutputKind::tensor) {
            strict_keys(o, path, {"kind", "shape"}, "a tensor output");
            if (auto it = o.find("shape"); it == o.end()) {
                missing(path, "shape", "A tensor output");
            } else if (!it->is_array()) {
                wrong(key_path(path, "shape"), *it, "a list of axis lengths");
            } else {
                for (std::size_t i = 0; i < it->size(); ++i) {
                    std::int64_t n = 0;
                    if (as_integer((*it)[i], n)) {
                        out.shape.push_back(n);
                    } else {
                        wrong(index_path(key_path(path, "shape"), i), (*it)[i], "an integer");
                    }
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

    // ---- pieces and their instances (design/programmable-graphics.md) ----------
    // The types, the keys, the required keys and the enumerations are checked
    // here; the names, the limits and the layouts are the checks of
    // programs/pieces.h, which the validator runs.

    Program instance_program(const json& o, const std::string& path) {
        Program p;
        static const std::vector<std::string> one_pass = {"wgsl", "entry", "inputs", "output", "limits", "workgroups"};
        for (const std::string& k : one_pass)
            if (o.find(k) != o.end())
                sink_.error("program.form_mixed",
                            "A program that pins a piece holds none of the keys of the one-pass form; " + quote(k) +
                                " belongs to that form.",
                            key_path(path, k));
        std::vector<std::string> allowed = {"id", "title", "piece", "bind", "panel", "params"};
        allowed.insert(allowed.end(), one_pass.begin(), one_pass.end());
        keys(o, path, allowed, "a program");
        p.id = required_string(o, "id", path, "A program");
        p.title = string(o, "title", path);
        if (auto it = o.find("piece"); it == o.end()) {
            missing(path, "piece", "A program with bind or panel");
        } else {
            p.piece = pin(*it, key_path(path, "piece"));
        }
        if (auto it = o.find("bind"); it != o.end()) {
            const std::string bp = key_path(path, "bind");
            if (table(*it, bp, bp)) {
                for (auto b = it->begin(); b != it->end(); ++b) {
                    if (b->is_string()) {
                        p.bind[b.key()] = b->get<std::string>();
                    } else {
                        wrong(key_path(bp, b.key()), *b, "a string, live:<tensor> or program:<id>/<resource>");
                    }
                }
            }
        }
        p.panel = string(o, "panel", path);
        p.params = params(o, path);
        p.x = extension(o, path);
        return p;
    }

    PieceRef pin(const json& o, const std::string& path) {
        PieceRef r;
        if (!table(o, path, path)) return r;
        strict_keys(o, path, {"name", "version", "hash", "from"}, "a pin");
        r.name = required_string(o, "name", path, "A pin");
        if (auto v = count(o, "version", path)) r.version = *v;
        if (auto h = string(o, "hash", path)) r.hash = *h;
        if (auto f = enumerated(o, "from", path, {"library", "workspace", "inline"}, "a source of pieces"))
            r.from = *piece_from_from_name(*f);
        return r;
    }

    // A list of strings at `key`, or nothing.
    std::vector<std::string> string_list(const json& o, const char* key, const std::string& path) {
        std::vector<std::string> out;
        auto it = o.find(key);
        if (it == o.end()) return out;
        if (!it->is_array()) {
            wrong(key_path(path, key), *it, "a list of strings");
            return out;
        }
        for (std::size_t i = 0; i < it->size(); ++i) {
            if ((*it)[i].is_string()) {
                out.push_back((*it)[i].get<std::string>());
            } else {
                wrong(index_path(key_path(path, key), i), (*it)[i], "a string");
            }
        }
        return out;
    }

    // The tables of the array at `key`, each decoded by `one`.
    template <class F>
    void tables(const json& o, const char* key, const std::string& path, F one) {
        auto it = o.find(key);
        if (it == o.end()) return;
        const std::string kp = key_path(path, key);
        if (!it->is_array()) {
            wrong(kp, *it, "a list of tables");
            return;
        }
        for (std::size_t i = 0; i < it->size(); ++i) {
            const std::string ip = index_path(kp, i);
            if (table((*it)[i], ip, ip)) one((*it)[i], ip);
        }
    }

    // `element`, or `struct` with `fields` written "name: type".
    Layout piece_layout(const json& o, const std::string& path) {
        Layout l;
        l.element = string(o, "element", path);
        l.type_name = string(o, "struct", path);
        const std::vector<std::string> fields = string_list(o, "fields", path);
        for (std::size_t i = 0; i < fields.size(); ++i) {
            const std::size_t colon = fields[i].find(": ");
            if (colon == std::string::npos) {
                sink_.error("spec.bad_value", "A field is written name: type, as \"weight: f32\".",
                            index_path(key_path(path, "fields"), i));
                continue;
            }
            l.fields.push_back(Field{fields[i].substr(0, colon), fields[i].substr(colon + 2)});
        }
        return l;
    }

    std::array<std::uint32_t, 3> workgroup_counts(const json& o, const std::string& path) {
        std::array<std::uint32_t, 3> w{1, 1, 1};
        auto it = o.find("workgroups");
        if (it == o.end()) return w;
        bool ok = it->is_array() && it->size() == 3;
        for (std::size_t i = 0; ok && i < 3; ++i) {
            std::int64_t v = 0;
            ok = as_integer((*it)[i], v) && v >= 0 && v <= 0xFFFFFFFFll;
            if (ok) w[i] = static_cast<std::uint32_t>(v);
        }
        if (!ok)
            sink_.error("spec.wrong_type", key_path(path, "workgroups") + " must be [x, y, z], three workgroup counts.",
                        key_path(path, "workgroups"));
        return w;
    }

    template <class E>
    std::optional<E> named(const json& o, const char* key, const std::string& path,
                           std::optional<E> (*from)(std::string_view), const std::vector<std::string>& values,
                           const char* what) {
        auto s = enumerated(o, key, path, values, what);
        return s ? from(*s) : std::nullopt;
    }

    DrawPass draw_pass(const json& o, const std::string& path) {
        DrawPass p;
        strict_keys(o, path,
                    {"kind", "entry", "workgroups", "writes", "vertex", "fragment", "target", "reads", "instances",
                     "vertices", "topology", "blend", "depth", "pick"},
                    "a pass of the draw half");
        if (o.find("kind") == o.end()) missing(path, "kind", "A pass of the draw half");
        if (auto k = named<PassKind>(o, "kind", path, pass_kind_from_name, {"compute", "render"}, "a kind of pass"))
            p.kind = *k;
        if (auto e = string(o, "entry", path)) p.entry = *e;
        p.workgroups = workgroup_counts(o, path);
        p.writes = string_list(o, "writes", path);
        p.vertex = string(o, "vertex", path);
        if (auto f = string(o, "fragment", path)) p.fragment = *f;
        if (auto t = string(o, "target", path)) p.target = *t;
        p.reads = string_list(o, "reads", path);
        if (auto it = o.find("instances"); it != o.end()) {
            if (it->is_string()) {
                p.instances_of = it->get<std::string>();
            } else if (auto n = count(o, "instances", path)) {
                p.instances = *n;
            }
        }
        p.vertices = count(o, "vertices", path);
        if (auto t = named<DrawTopology>(o, "topology", path, draw_topology_from_name,
                                         {"triangle_list", "triangle_strip", "line_list", "line_strip", "point_list"},
                                         "a topology"))
            p.topology = *t;
        p.blend = named<DrawBlend>(o, "blend", path, draw_blend_from_name, {"opaque", "alpha", "additive"}, "a blend");
        p.depth = named<DrawDepth>(o, "depth", path, draw_depth_from_name, {"none", "test", "write"}, "a depth");
        if (auto b = boolean(o, "pick", path)) p.pick = *b;
        return p;
    }

    Draw draw_half(const json& o, const std::string& path) {
        Draw d;
        strict_keys(o, path, {"camera", "buffers", "targets", "passes", "pick", "text", "frame_ms", "memory_mb"},
                    "the draw table");
        if (auto c = named<CameraMode>(o, "camera", path, camera_mode_from_name, {"orbit", "pan_zoom", "none"},
                                       "a camera"))
            d.camera = *c;
        tables(o, "buffers", path, [&](const json& b, const std::string& bp) {
            strict_keys(b, bp, {"id", "element", "struct", "fields", "count", "reset"}, "a draw buffer");
            DrawBuffer buf;
            buf.id = required_string(b, "id", bp, "A draw buffer");
            buf.layout = piece_layout(b, bp);
            if (auto n = count(b, "count", bp)) buf.count = *n;
            if (auto r = named<BufferReset>(b, "reset", bp, buffer_reset_from_name, {"install", "cursor", "resource"},
                                            "a reset"))
                buf.reset = *r;
            d.buffers.push_back(std::move(buf));
        });
        tables(o, "targets", path, [&](const json& t, const std::string& tp) {
            strict_keys(t, tp, {"id", "scale", "clear"}, "a target");
            DrawTarget target;
            target.id = required_string(t, "id", tp, "A target");
            if (auto s = number(t, "scale", tp)) target.scale = static_cast<float>(*s);
            if (auto it = t.find("clear"); it != t.end()) {
                bool ok = it->is_array() && it->size() == 4;
                for (std::size_t i = 0; ok && i < 4; ++i) {
                    ok = (*it)[i].is_number();
                    if (ok) target.clear[i] = (*it)[i].get<float>();
                }
                if (!ok)
                    sink_.error("spec.wrong_type", key_path(tp, "clear") + " must be [r, g, b, a], four numbers.",
                                key_path(tp, "clear"));
            }
            d.targets.push_back(std::move(target));
        });
        if (o.find("passes") == o.end()) missing(path, "passes", "The draw table");
        tables(o, "passes", path, [&](const json& p, const std::string& pp) { d.passes.push_back(draw_pass(p, pp)); });
        if (auto it = o.find("pick"); it != o.end()) {
            const std::string pp = key_path(path, "pick");
            if (table(*it, pp, pp)) {
                strict_keys(*it, pp, {"resource", "fields"}, "the item map");
                PickMap m;
                m.resource = required_string(*it, "resource", pp, "The item map");
                m.fields = string_list(*it, "fields", pp);
                d.pick = std::move(m);
            }
        }
        if (auto it = o.find("text"); it != o.end()) {
            const std::string tp = key_path(path, "text");
            if (table(*it, tp, tp)) {
                strict_keys(*it, tp, {"anchors", "position", "string", "priority", "strings"}, "the text table");
                TextTable t;
                t.anchors = required_string(*it, "anchors", tp, "The text table");
                if (auto v = string(*it, "position", tp)) t.position = *v;
                if (auto v = string(*it, "string", tp)) t.string = *v;
                t.priority = string(*it, "priority", tp);
                t.strings = string_list(*it, "strings", tp);
                d.text = std::move(t);
            }
        }
        d.frame_ms = number(o, "frame_ms", path);
        d.memory_mb = count(o, "memory_mb", path);
        return d;
    }

    // One piece: the keys of piece.toml, and for an inline piece and the
    // commands its WGSL in `files` under common, derive and draw.
    Piece piece(const json& o, const std::string& path) {
        Piece p;
        if (!table(o, path, path)) return p;
        strict_keys(o, path,
                    {"name", "version", "description", "inputs", "params", "resources", "imports", "depends", "derive",
                     "draw", "sample", "hash", "files"},
                    "a piece");
        p.name = required_string(o, "name", path, "A piece");
        if (auto v = count(o, "version", path)) p.version = *v;
        if (auto v = string(o, "description", path)) p.description = *v;
        tables(o, "inputs", path, [&](const json& t, const std::string& ip) {
            strict_keys(t, ip,
                        {"name", "kind", "dtype", "rank_min", "rank_max", "element", "struct", "fields", "title",
                         "optional"},
                        "an input");
            PieceInput in;
            in.name = required_string(t, "name", ip, "An input");
            if (t.find("kind") == t.end()) missing(ip, "kind", "An input");
            if (auto k = named<InputKind>(t, "kind", ip, input_kind_from_name, {"tensor", "buffer"}, "a kind of input"))
                in.kind = *k;
            if (auto d = enumerated(t, "dtype", ip, {"f32", "i32", "u32"}, "a dtype")) in.dtype = *d;
            if (auto v = count(t, "rank_min", ip)) in.rank_min = *v;
            if (auto v = count(t, "rank_max", ip)) in.rank_max = *v;
            in.layout = piece_layout(t, ip);
            in.title = string(t, "title", ip);
            if (auto v = boolean(t, "optional", ip)) in.optional = *v;
            p.inputs.push_back(std::move(in));
        });
        tables(o, "params", path, [&](const json& t, const std::string& pp) {
            strict_keys(t, pp, {"name", "type", "default", "min", "max", "title"}, "a parameter");
            PieceParam param;
            param.name = required_string(t, "name", pp, "A parameter");
            if (auto ty = named<ParamType>(t, "type", pp, param_type_from_name, {"f32", "i32", "u32"}, "a type"))
                param.type = *ty;
            if (auto v = number(t, "default", pp)) param.default_value = *v;
            param.min = number(t, "min", pp);
            param.max = number(t, "max", pp);
            param.title = string(t, "title", pp);
            p.params.push_back(std::move(param));
        });
        tables(o, "resources", path, [&](const json& t, const std::string& rp) {
            strict_keys(t, rp,
                        {"id", "kind", "element", "struct", "fields", "count", "counted", "width", "height", "publish",
                         "clear", "keep"},
                        "a resource");
            Resource r;
            r.id = required_string(t, "id", rp, "A resource");
            if (t.find("kind") == t.end()) missing(rp, "kind", "A resource");
            if (auto k = named<ResourceKind>(t, "kind", rp, resource_kind_from_name, {"buffer", "texture"},
                                             "a kind of resource"))
                r.kind = *k;
            r.layout = piece_layout(t, rp);
            if (auto v = count(t, "count", rp)) r.count = *v;
            if (auto v = boolean(t, "counted", rp)) r.counted = *v;
            if (auto v = count(t, "width", rp)) r.width = *v;
            if (auto v = count(t, "height", rp)) r.height = *v;
            if (auto v = boolean(t, "publish", rp)) r.publish = *v;
            if (auto v = boolean(t, "clear", rp)) r.clear = *v;
            r.keep = count(t, "keep", rp);
            p.resources.push_back(std::move(r));
        });
        p.imports = string_list(o, "imports", path);
        tables(o, "depends", path, [&](const json& t, const std::string& dp) {
            strict_keys(t, dp, {"name", "version"}, "a dependency");
            PieceDependency d;
            d.name = required_string(t, "name", dp, "A dependency");
            if (auto v = count(t, "version", dp)) d.version = *v;
            p.depends.push_back(std::move(d));
        });
        if (auto it = o.find("derive"); it != o.end()) {
            const std::string dp = key_path(path, "derive");
            if (table(*it, dp, dp)) {
                strict_keys(*it, dp, {"passes", "time_ms", "memory_mb", "every"}, "the derive table");
                Derive d;
                if (it->find("passes") == it->end()) missing(dp, "passes", "The derive table");
                tables(*it, "passes", dp, [&](const json& t, const std::string& pp) {
                    strict_keys(t, pp, {"entry", "workgroups", "writes"}, "a pass of the derive half");
                    DerivePass pass;
                    pass.entry = required_string(t, "entry", pp, "A pass of the derive half");
                    pass.workgroups = workgroup_counts(t, pp);
                    pass.writes = string_list(t, "writes", pp);
                    d.passes.push_back(std::move(pass));
                });
                d.time_ms = number(*it, "time_ms", dp);
                d.memory_mb = count(*it, "memory_mb", dp);
                d.every = count(*it, "every", dp);
                p.derive = std::move(d);
            }
        }
        if (auto it = o.find("draw"); it != o.end()) {
            const std::string dp = key_path(path, "draw");
            if (table(*it, dp, dp)) p.draw = draw_half(*it, dp);
        }
        tables(o, "sample", path, [&](const json& t, const std::string& sp) {
            strict_keys(t, sp, {"input", "shape"}, "a sample");
            const std::string input = required_string(t, "input", sp, "A sample");
            SampleInput s;
            s.file = "sample/" + input + ".bin";
            if (auto it = t.find("shape"); it == t.end()) {
                missing(sp, "shape", "A sample");
            } else if (!it->is_array()) {
                wrong(key_path(sp, "shape"), *it, "a list of axis lengths");
            } else {
                for (std::size_t i = 0; i < it->size(); ++i) {
                    std::int64_t v = 0;
                    if (as_integer((*it)[i], v) && v >= 0 && v <= 0xFFFFFFFFll) {
                        s.shape.push_back(static_cast<std::uint32_t>(v));
                    } else {
                        wrong(index_path(key_path(sp, "shape"), i), (*it)[i], "an axis length");
                    }
                }
            }
            if (!p.sample.emplace(input, std::move(s)).second)
                sink_.error("spec.duplicate_id", "Two samples name the input " + quote(input) + ".", sp);
        });
        if (auto v = string(o, "hash", path)) p.hash = *v;
        if (auto it = o.find("files"); it != o.end()) {
            const std::string fp = key_path(path, "files");
            if (table(*it, fp, fp)) {
                strict_keys(*it, fp, {"common", "derive", "draw"}, "the files of a piece");
                for (const char* part : {"common", "derive", "draw"})
                    if (auto text = string(*it, part, fp)) p.files[std::string(part) + ".wgsl"] = *text;
            }
        }
        return p;
    }

    Stage stage(const json& o, const std::string& path) {
        Stage st;
        if (!table(o, path, path)) return st;
        keys(o, path, {"panel", "entity", "kind", "wgsl", "params"}, "a stage");
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
        st.params = params(o, path);
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
        bool kind_known = false;
        if (auto kind = string(o, "kind", path)) {
            if (auto k = control_kind_from_name(*kind)) {
                c.kind = *k;
                kind_known = true;
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
        } else if (o.find("bind") == o.end() && !(kind_known && c.kind == ControlKind::readout)) {
            missing(path, "bind", "A control");  // a readout binds nothing
        }
        if (auto it = o.find("options"); it != o.end()) {
            const std::string op = key_path(path, "options");
            if (table(*it, op, op)) {
                const json& opts = *it;
                ControlOptions& out = c.options;
                for (auto k = opts.begin(); k != opts.end(); ++k) {
                    if (k.key() == "x" || contains(control_option_names(), k.key())) continue;
                    sink_.error("spec.unknown_option",
                                quote(k.key()) + " is not an option of a control; its options are " +
                                    join(control_option_names()) + ".",
                                key_path(op, k.key()));
                }
                out.min = or_reference(opts, "min", op, out.references, [&] { return number(opts, "min", op); });
                out.max = or_reference(opts, "max", op, out.references, [&] { return number(opts, "max", op); });
                out.step = number(opts, "step", op);
                out.log = boolean(opts, "log", op);
                // A number is the start value of a slider; any other value is a
                // range, a boolean, a text or a list, kept as written.
                if (auto v = opts.find("value"); v != opts.end()) {
                    if (v->is_number()) {
                        out.value = v->get<double>();
                    } else {
                        out.value_other = *v;
                    }
                }
                if (auto ch = opts.find("choices"); ch != opts.end()) out.choices = choices(*ch, key_path(op, "choices"));
                out.choices_from = string(opts, "choices_from", op);
                out.multiple = boolean(opts, "multiple", op);
                out.placeholder = string(opts, "placeholder", op);
                out.max_length = integer(opts, "max_length", op);
                out.min_span = number(opts, "min_span", op);
                out.follow = boolean(opts, "follow", op);
                out.speed = number(opts, "speed", op);
                out.keys = boolean(opts, "keys", op);
                out.bound_text = bound_text(opts, "format", op, "The options of a readout");
                out.x = extension(opts, op);
            }
        }
        c.x = extension(o, path);
        return c;
    }

    // The choices of a picker or a segmented group: values, or tables
    // { value, label, key }. The count and the keys are the validator's.
    std::vector<Choice> choices(const json& list, const std::string& path) {
        std::vector<Choice> out;
        if (!list.is_array()) {
            wrong(path, list, "a list of choices");
            return out;
        }
        for (std::size_t i = 0; i < list.size(); ++i) {
            const json& item = list[i];
            const std::string ip = index_path(path, i);
            Choice c;
            if (item.is_object()) {
                c.table = true;
                strict_keys(item, ip, {"value", "label", "key"}, "a choice");
                if (auto v = item.find("value"); v != item.end()) {
                    c.value = *v;
                } else {
                    missing(ip, "value", "A choice");
                }
                c.label = string(item, "label", ip);
                c.key = string(item, "key", ip);
            } else {
                c.value = item;
            }
            out.push_back(std::move(c));
        }
        if (list.empty())
            sink_.error("spec.bad_value", path + " lists no choice; leave it out to give the control none.", path);
        return out;
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
        if (version_ >= kInstrumentVersion) {
            keys(o, path, {"id", "catalog", "sql", "from", "transform", "history"}, "a source");
        } else {
            keys(o, path, {"id", "catalog", "sql", "from", "transform"}, "a source");
        }
        if (auto id = string(o, "id", path)) {
            s.id = *id;
        } else if (o.find("id") == o.end()) {
            missing(path, "id", "A source");
        }
        s.catalog = string(o, "catalog", path);
        s.sql = string(o, "sql", path);
        s.from = string(o, "from", path);
        if (version_ >= kInstrumentVersion) s.history = integer(o, "history", path);
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
        const bool v1_op = transform_op_from_string(*op, t.op) &&
                           (t.op == TransformOp::window || t.op == TransformOp::surface || t.op == TransformOp::path);
        if (!transform_op_from_string(*op, t.op) || (v1_op && version_ < kInstrumentVersion)) {
            sink_.error("spec.unknown_transform",
                        quote(*op) + " is not a transform; the transforms are " +
                            (version_ >= kInstrumentVersion ? "slice, reduce, normalise, window, surface and path."
                                                            : "slice, reduce and normalise."),
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
        } else if (t.op == TransformOp::slice || t.op == TransformOp::reduce) {
            missing(path, "axis", std::string("A ") + to_string(t.op) + " transform");
        }
        auto integer_step = [&](const char* key) {
            return or_reference(o, key, path, t.references, [&] { return integer(o, key, path); });
        };
        if (t.op == TransformOp::slice) {
            t.index = integer_step("index");
            t.start = integer_step("start");
            t.stop = integer_step("stop");
            t.step = integer_step("step");
        }
        if (t.op == TransformOp::window) {
            t.start = integer_step("start");
            t.stop = integer_step("stop");
            t.steps = integer_step("steps");
        }
        if (t.op == TransformOp::surface) {
            t.max_cells = integer(o, "max_cells", path);
            t.pool = enumerated(o, "pool", path, pool_modes(), "a pooling");
        }
        if (t.op == TransformOp::path) {
            t.max_points = integer(o, "max_points", path);
            if (auto it = o.find("columns"); it != o.end()) {
                const std::string cp = key_path(path, "columns");
                if (!it->is_array()) {
                    wrong(cp, *it, "a list of column indices");
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i) {
                        std::int64_t n = 0;
                        if (as_integer((*it)[i], n)) {
                            t.columns.push_back(n);
                        } else {
                            wrong(index_path(cp, i), (*it)[i], "an integer");
                        }
                    }
                    if (it->empty())
                        sink_.error("spec.bad_value", cp + " lists no column; leave it out to take the default.", cp);
                }
            }
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
        if (version_ >= kInstrumentVersion) {
            keys(o, path, {"id", "primitive", "title", "bind", "options", "entity", "annotation"}, "a panel");
        } else {
            keys(o, path, {"id", "primitive", "title", "bind", "options"}, "a panel");
        }
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
                        Reference r;
                        switch (reference(*s, sp, r)) {
                            case Ref::read: value.reference = std::move(r); break;
                            case Ref::bad: continue;
                            case Ref::none:
                                wrong(sp, *s, "a binding string or a list of them");
                                continue;
                        }
                    }
                    p.bind[s.key()] = std::move(value);
                }
            }
        }

        if (auto it = o.find("options"); it != o.end()) {
            const std::string op = key_path(path, "options");
            if (table(*it, op, op)) {
                const json& opts = *it;
                std::vector<std::string> allowed = options_of(p.primitive);
                if (version_ < kInstrumentVersion)
                    allowed.erase(std::remove_if(allowed.begin(), allowed.end(),
                                                 [&](const std::string& n) { return version_one_option(p.primitive, n); }),
                                  allowed.end());
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
                for (const char* key : {"x_range", "y_range", "range"}) {
                    if (!has(key)) continue;
                    auto value = or_reference(opts, key, op, out.references, [&] { return range(opts, key, op); });
                    const std::string name = key;
                    (name == "x_range" ? out.x_range : name == "y_range" ? out.y_range : out.range) = value;
                }
                if (has("x_log")) out.x_log = boolean(opts, "x_log", op);
                if (has("y_log")) out.y_log = boolean(opts, "y_log", op);
                if (has("colormap"))
                    out.colormap = or_reference(opts, "colormap", op, out.references, [&] {
                        return enumerated(opts, "colormap", op, colormaps(), "a colormap");
                    });
                if (has("bins")) out.bins = integer(opts, "bins", op);
                if (has("text")) out.text = string(opts, "text", op);
                if (has("up")) out.scene.up = enumerated(opts, "up", op, scene_up_axes(), "an axis");
                if (has("view_camera")) out.scene.view_camera = view_camera(opts["view_camera"], key_path(op, "view_camera"));
                if (has("include")) out.scene.include = patterns(opts, "include", op);
                if (has("exclude")) out.scene.exclude = patterns(opts, "exclude", op);
                if (has("labels")) out.scene.labels = boolean(opts, "labels", op);
                if (has("axes")) out.scene.axes = boolean(opts, "axes", op);
                if (has("scale_bar")) out.scene.scale_bar = boolean(opts, "scale_bar", op);
                if (has("legend")) {
                    const json& l = opts["legend"];
                    const std::string lp = key_path(op, "legend");
                    if (l.is_boolean() || (l.is_string() && l.get<std::string>() == "auto")) {
                        out.legend = l;
                    } else if (l.is_string()) {
                        sink_.error("spec.bad_value", lp + " must be \"auto\", true or false.", lp);
                    } else {
                        wrong(lp, l, "\"auto\", true or false");
                    }
                }
                if (has("box")) {
                    const json& b = opts["box"];
                    const std::string bp = key_path(op, "box");
                    bool numbers_ok = b.is_array() && b.size() == 3;
                    for (std::size_t i = 0; numbers_ok && i < 3; ++i) numbers_ok = b[i].is_number();
                    if (numbers_ok || (b.is_string() && b.get<std::string>() == "auto")) {
                        out.box = b;
                    } else if (b.is_string()) {
                        sink_.error("spec.bad_value", bp + " must be \"auto\" or [x, y, z], three numbers.", bp);
                    } else {
                        wrong(bp, b, "\"auto\" or [x, y, z], three numbers");
                    }
                }
                if (has("views")) out.views = views(opts["views"], key_path(op, "views"));
                if (has("select")) out.select = select(opts["select"], key_path(op, "select"));
                if (has("format") || has("fields")) out.bound_text = bound_text(opts, "format", op, "The options of a text panel");
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
        if (version_ >= kInstrumentVersion) {
            if (auto it = o.find("entity"); it != o.end()) {
                const std::string ep = key_path(path, "entity");
                if (!it->is_array()) {
                    sink_.error("spec.wrong_type",
                                ep + " must be an array of tables; write each one as [[panel.entity]].", ep);
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i)
                        p.entities.push_back(entity((*it)[i], index_path(ep, i)));
                }
            }
            if (auto it = o.find("annotation"); it != o.end()) {
                const std::string ap = key_path(path, "annotation");
                if (!it->is_array()) {
                    sink_.error("spec.wrong_type",
                                ap + " must be an array of tables; write each one as [[panel.annotation]].", ap);
                } else {
                    for (std::size_t i = 0; i < it->size(); ++i)
                        p.annotations.push_back(annotation((*it)[i], index_path(ap, i)));
                }
            }
        }
        return p;
    }

    // The option `views`: at most 16 saved camera views, each with all five keys.
    std::vector<SavedView> views(const json& list, const std::string& path) {
        std::vector<SavedView> out;
        if (!list.is_array()) {
            wrong(path, list, "a list of saved views");
            return out;
        }
        for (std::size_t i = 0; i < list.size(); ++i) {
            const std::string ip = index_path(path, i);
            const json& item = list[i];
            SavedView v;
            if (table(item, ip, ip)) {
                strict_keys(item, ip, {"name", "target", "distance", "azimuth", "elevation"}, "a saved view");
                v.name = required_string(item, "name", ip, "A saved view");
                for (const char* key : {"target", "distance", "azimuth", "elevation"})
                    if (item.find(key) == item.end()) missing(ip, key, "A saved view");
                numbers(item, "target", ip, v.target);
                if (auto d = number(item, "distance", ip)) v.distance = *d;
                if (auto a = number(item, "azimuth", ip)) v.azimuth = *a;
                if (auto e = number(item, "elevation", ip)) v.elevation = *e;
            }
            out.push_back(std::move(v));
        }
        if (list.empty())
            sink_.error("spec.bad_value", path + " lists no view; leave it out to save none.", path);
        return out;
    }

    // The option `select`: a state variable and the pick field that it takes.
    std::optional<StateWrite> select(const json& v, const std::string& path) {
        if (!table(v, path, path)) return std::nullopt;
        strict_keys(v, path, {"state", "field", "entity"}, "the option select");
        StateWrite w;
        w.state = required_string(v, "state", path, "The option select");
        w.field = required_string(v, "field", path, "The option select");
        w.entity = string(v, "entity", path);
        return w;
    }

    // ---- state variables, entities and annotations (spec version 1) --------------
    // The types, the keys, the required keys and the enumerations are checked
    // here; the values, the bounds and the references are the validator's.

    StateVar state(const json& o, const std::string& path) {
        StateVar s;
        if (!table(o, path, path)) return s;
        keys(o, path, {"id", "title", "type", "default", "min", "max", "choices"}, "a state variable");
        s.id = required_string(o, "id", path, "A state variable");
        s.title = string(o, "title", path);
        if (auto type = string(o, "type", path)) {
            if (auto t = state_type_from_name(*type)) {
                s.type = *t;
            } else {
                sink_.error("spec.bad_value",
                            quote(*type) + " is not a type of state variable; the types are number, integer, "
                                           "boolean, text, range and list.",
                            key_path(path, "type"));
            }
        } else if (o.find("type") == o.end()) {
            missing(path, "type", "A state variable");
        }
        if (auto it = o.find("default"); it != o.end()) s.default_value = *it;
        s.min = number(o, "min", path);
        s.max = number(o, "max", path);
        if (auto it = o.find("choices"); it != o.end()) {
            const std::string cp = key_path(path, "choices");
            if (!it->is_array()) {
                wrong(cp, *it, "a list of values");
            } else {
                for (const json& v : *it) s.choices.push_back(v);
                if (it->empty())
                    sink_.error("spec.bad_value", cp + " lists no value; leave it out to allow any.", cp);
            }
        }
        s.x = extension(o, path);
        return s;
    }

    Entity entity(const json& o, const std::string& path) {
        Entity e;
        if (!table(o, path, path)) return e;
        keys(o, path, {"path", "draw", "title", "bind", "options", "pose"}, "an entity");
        e.path = required_string(o, "path", path, "An entity");
        if (auto draw = string(o, "draw", path)) {
            if (auto d = draw_kind_from_name(*draw)) {
                e.draw = *d;
            } else {
                sink_.error("spec.bad_value",
                            quote(*draw) + " is not a draw kind; the kinds are points, lines and surface.",
                            key_path(path, "draw"));
            }
        } else if (o.find("draw") == o.end()) {
            missing(path, "draw", "An entity");
        }
        e.title = string(o, "title", path);
        if (auto it = o.find("bind"); it == o.end()) {
            missing(path, "bind", "An entity");
        } else if (it->is_string()) {
            e.source = it->get<std::string>();
        } else if (it->is_object() && !looks_like_reference(*it)) {
            const std::string bp = key_path(path, "bind");
            for (auto slot = it->begin(); slot != it->end(); ++slot) {
                const std::string sp = key_path(bp, slot.key());
                Reference r;
                if (slot->is_string()) {
                    e.slots[slot.key()] = slot->get<std::string>();
                } else {
                    switch (reference(*slot, sp, r)) {
                        case Ref::read: e.slots[slot.key()] = std::move(r); break;
                        case Ref::bad: break;
                        case Ref::none: wrong(sp, *slot, "a binding string or a reference"); break;
                    }
                }
            }
        } else {
            wrong(key_path(path, "bind"), *it, "a source id or a table of slots");
        }
        if (auto it = o.find("options"); it != o.end()) {
            const std::string op = key_path(path, "options");
            if (table(*it, op, op) && !it->empty()) e.options = *it;
        }
        if (auto it = o.find("pose"); it != o.end()) {
            const std::string pp = key_path(path, "pose");
            if (table(*it, pp, pp)) {
                strict_keys(*it, pp, {"position", "rotation", "scale"}, "a pose");
                Pose pose;
                numbers(*it, "position", pp, pose.position);
                numbers(*it, "rotation", pp, pose.rotation);
                numbers(*it, "scale", pp, pose.scale);
                e.pose = pose;
            }
        }
        e.x = extension(o, path);
        return e;
    }

    Annotation annotation(const json& o, const std::string& path) {
        Annotation a;
        if (!table(o, path, path)) return a;
        keys(o, path, {"id", "kind", "axis", "value", "range", "at", "text", "fields", "leader", "emphasis"},
             "an annotation");
        a.id = required_string(o, "id", path, "An annotation");
        if (auto kind = string(o, "kind", path)) {
            if (auto k = annotation_kind_from_name(*kind)) {
                a.kind = *k;
            } else {
                sink_.error("spec.bad_value",
                            quote(*kind) + " is not a kind of annotation; the kinds are label, threshold, band and "
                                           "marker.",
                            key_path(path, "kind"));
            }
        } else if (o.find("kind") == o.end()) {
            missing(path, "kind", "An annotation");
        }
        a.axis = string(o, "axis", path);
        if (auto it = o.find("value"); it != o.end()) a.value = *it;
        if (auto it = o.find("range"); it != o.end()) a.range = *it;
        if (auto it = o.find("at"); it != o.end()) a.at = *it;
        a.text = bound_text(o, "text", path, "An annotation");
        a.leader = boolean(o, "leader", path);
        a.emphasis = string(o, "emphasis", path);
        a.x = extension(o, path);
        return a;
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
        if (version_ >= kInstrumentVersion && v.find("panel") != v.end()) {
            // A leaf table { panel, min } (spec version 1).
            if (has_split || has_tabs) {
                sink_.error("spec.bad_layout", path + " has panel together with split or tabs; a layout table has one.",
                            path);
                return n;
            }
            keys(v, path, {"panel", "min"}, "a layout leaf");
            n.kind = LayoutNode::Kind::panel;
            if (auto id = string(v, "panel", path)) n.panel = *id;
            if (auto it = v.find("min"); it != v.end()) {
                const std::string mp = key_path(path, "min");
                std::int64_t columns = 0, lines = 0;
                if (it->is_array() && it->size() == 2 && as_integer((*it)[0], columns) &&
                    as_integer((*it)[1], lines)) {
                    n.min = std::array<std::int64_t, 2>{columns, lines};
                } else {
                    sink_.error("spec.wrong_type", mp + " must be [columns, lines], two integers.", mp);
                }
            }
            n.x = extension(v, path);
            return n;
        }
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
            if (version_ >= kInstrumentVersion) {
                keys(v, path, {"split", "ratio", "shares", "title", "children"}, "a split");
            } else {
                keys(v, path, {"split", "ratio", "title", "children"}, "a split");
            }
            static const std::vector<std::string> directions = {"horizontal", "vertical"};
            if (auto d = enumerated(v, "split", path, directions, "a split direction"))
                split_from_string(*d, n.split);
            n.ratio = number(v, "ratio", path);
            if (auto sh = v.find("shares"); sh != v.end() && version_ >= kInstrumentVersion) {
                const std::string sp = key_path(path, "shares");
                if (!sh->is_array()) {
                    wrong(sp, *sh, "a list of numbers");
                } else {
                    for (std::size_t i = 0; i < sh->size(); ++i) {
                        if ((*sh)[i].is_number()) {
                            n.shares.push_back((*sh)[i].get<double>());
                        } else {
                            wrong(index_path(sp, i), (*sh)[i], "a number");
                        }
                    }
                    if (sh->empty())
                        sink_.error("spec.bad_value", sp + " lists no share; leave it out to split by ratio.", sp);
                }
            }
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

Piece decode_piece(const json& item, const std::string& path, Sink& sink) { return Decoder(sink).piece(item, path); }

StateVar decode_state(const json& item, const std::string& path, Sink& sink) {
    return Decoder(sink).state(item, path);
}

Entity decode_entity(const json& item, const std::string& path, Sink& sink) {
    return Decoder(sink).entity(item, path);
}

Annotation decode_annotation(const json& item, const std::string& path, Sink& sink) {
    return Decoder(sink).annotation(item, path);
}

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

PieceParse piece_from_json_value(const json& item, const std::string& path) {
    Decoded<Piece> d = decode_item<Piece>(item, path, 2, detail::decode_piece);
    return PieceParse{std::move(d.value), std::move(d.diagnostics)};
}

Decoded<StateVar> state_from_json(const json& item, const std::string& path) {
    return decode_item<StateVar>(item, path, 2, detail::decode_state);
}

Decoded<Entity> entity_from_json(const json& item, const std::string& path) {
    return decode_item<Entity>(item, path, 3, detail::decode_entity);
}

Decoded<Annotation> annotation_from_json(const json& item, const std::string& path) {
    return decode_item<Annotation>(item, path, 3, detail::decode_annotation);
}

// ---- writing ----------------------------------------------------------------

namespace {

// The table of `fields`: a source id, or the table of a reference.
json fields_json(const BoundText& t) {
    json j = json::object();
    for (const auto& [name, field] : t.fields) {
        if (const auto* source = std::get_if<std::string>(&field)) {
            j[name] = *source;
        } else {
            j[name] = reference_json(std::get<Reference>(field));
        }
    }
    return j;
}

// The keys of a bound text: its format under `format_key`, and `fields`
// when there are any.
void write_bound_text(json& j, const char* format_key, const BoundText& t) {
    j[format_key] = t.format;
    if (!t.fields.empty()) j["fields"] = fields_json(t);
}

json params_json(const Params& params) {
    json j = json::object();
    for (const auto& [name, value] : params) {
        if (const auto* number = std::get_if<double>(&value)) {
            j[name] = *number;
        } else {
            j[name] = reference_json(std::get<Reference>(value));
        }
    }
    return j;
}

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
    if (t.steps) j["steps"] = *t.steps;
    if (t.max_cells) j["max_cells"] = *t.max_cells;
    if (t.pool) j["pool"] = *t.pool;
    if (!t.columns.empty()) j["columns"] = t.columns;
    if (t.max_points) j["max_points"] = *t.max_points;
    for (const auto& [key, ref] : t.references) j[key] = reference_json(ref);
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
    if (!o.legend.is_null()) j["legend"] = o.legend;
    if (!o.box.is_null()) j["box"] = o.box;
    if (!o.views.empty()) {
        json list = json::array();
        for (const SavedView& v : o.views)
            list.push_back(json{{"name", v.name},
                                {"target", json::array({v.target[0], v.target[1], v.target[2]})},
                                {"distance", v.distance},
                                {"azimuth", v.azimuth},
                                {"elevation", v.elevation}});
        j["views"] = std::move(list);
    }
    if (o.select) {
        json select = json{{"state", o.select->state}, {"field", o.select->field}};
        if (o.select->entity) select["entity"] = *o.select->entity;
        j["select"] = std::move(select);
    }
    if (o.bound_text) write_bound_text(j, "format", *o.bound_text);
    for (const auto& [key, ref] : o.references) j[key] = reference_json(ref);
    if (!o.x.is_null()) j["x"] = o.x;
    return j;
}

json control_options_json(const ControlOptions& o) {
    json j = json::object();
    if (o.min) j["min"] = *o.min;
    if (o.max) j["max"] = *o.max;
    if (o.step) j["step"] = *o.step;
    if (o.log) j["log"] = *o.log;
    if (o.value) {
        j["value"] = *o.value;
    } else if (!o.value_other.is_null()) {
        j["value"] = o.value_other;
    }
    if (!o.choices.empty()) {
        json list = json::array();
        for (const Choice& c : o.choices) {
            if (!c.table) {
                list.push_back(c.value);
                continue;
            }
            json item = json{{"value", c.value}};
            if (c.label) item["label"] = *c.label;
            if (c.key) item["key"] = *c.key;
            list.push_back(std::move(item));
        }
        j["choices"] = std::move(list);
    }
    if (o.choices_from) j["choices_from"] = *o.choices_from;
    if (o.multiple) j["multiple"] = *o.multiple;
    if (o.placeholder) j["placeholder"] = *o.placeholder;
    if (o.max_length) j["max_length"] = *o.max_length;
    if (o.min_span) j["min_span"] = *o.min_span;
    if (o.follow) j["follow"] = *o.follow;
    if (o.speed) j["speed"] = *o.speed;
    if (o.keys) j["keys"] = *o.keys;
    if (o.bound_text) write_bound_text(j, "format", *o.bound_text);
    for (const auto& [key, ref] : o.references) j[key] = reference_json(ref);
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
    if (s.history) j["history"] = *s.history;
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
            if (value.reference) {
                bind[slot] = reference_json(*value.reference);
            } else if (!value.list && value.bindings.size() == 1) {
                bind[slot] = value.bindings[0];
            } else {
                bind[slot] = value.bindings;
            }
        }
        j["bind"] = std::move(bind);
    }
    if (!p.options.empty()) j["options"] = options_json(p.options);
    if (!p.entities.empty()) {
        json list = json::array();
        for (const Entity& e : p.entities) list.push_back(to_json(e));
        j["entity"] = std::move(list);
    }
    if (!p.annotations.empty()) {
        json list = json::array();
        for (const Annotation& a : p.annotations) list.push_back(to_json(a));
        j["annotation"] = std::move(list);
    }
    if (!p.x.is_null()) j["x"] = p.x;
    return j;
}

json to_json(const StateVar& s) {
    json j = json::object();
    j["id"] = s.id;
    if (s.title) j["title"] = *s.title;
    j["type"] = state_type_name(s.type);
    if (!s.default_value.is_null()) j["default"] = s.default_value;
    if (s.min) j["min"] = *s.min;
    if (s.max) j["max"] = *s.max;
    if (!s.choices.empty()) j["choices"] = s.choices;
    if (!s.x.is_null()) j["x"] = s.x;
    return j;
}

json to_json(const Entity& e) {
    json j = json::object();
    j["path"] = e.path;
    j["draw"] = draw_kind_name(e.draw);
    if (e.title) j["title"] = *e.title;
    if (e.source) {
        j["bind"] = *e.source;
    } else {
        json bind = json::object();
        for (const auto& [slot, value] : e.slots) {
            if (const auto* source = std::get_if<std::string>(&value)) {
                bind[slot] = *source;
            } else {
                bind[slot] = reference_json(std::get<Reference>(value));
            }
        }
        j["bind"] = std::move(bind);
    }
    if (!e.options.is_null()) j["options"] = e.options;
    if (e.pose) {
        j["pose"] = json{{"position", json::array({e.pose->position[0], e.pose->position[1], e.pose->position[2]})},
                         {"rotation", json::array({e.pose->rotation[0], e.pose->rotation[1], e.pose->rotation[2],
                                                   e.pose->rotation[3]})},
                         {"scale", json::array({e.pose->scale[0], e.pose->scale[1], e.pose->scale[2]})}};
    }
    if (!e.x.is_null()) j["x"] = e.x;
    return j;
}

json to_json(const Annotation& a) {
    json j = json::object();
    j["id"] = a.id;
    j["kind"] = annotation_kind_name(a.kind);
    if (a.axis) j["axis"] = *a.axis;
    if (!a.value.is_null()) j["value"] = a.value;
    if (!a.range.is_null()) j["range"] = a.range;
    if (!a.at.is_null()) j["at"] = a.at;
    if (a.text) write_bound_text(j, "text", *a.text);
    if (a.leader) j["leader"] = *a.leader;
    if (a.emphasis) j["emphasis"] = *a.emphasis;
    if (!a.x.is_null()) j["x"] = a.x;
    return j;
}

json to_json(const Control& c) {
    json j = json::object();
    j["id"] = c.id;
    j["kind"] = control_kind_name(c.kind);
    if (c.title) j["title"] = *c.title;
    if (!c.bind.empty() || c.kind != ControlKind::readout) j["bind"] = c.bind;  // a readout binds nothing
    if (!c.options.empty()) j["options"] = control_options_json(c.options);
    if (!c.x.is_null()) j["x"] = c.x;
    return j;
}

json to_json(const PieceRef& r) {
    json j = {{"name", r.name}, {"version", r.version}, {"from", piece_from_name(r.from)}};
    if (!r.hash.empty()) j["hash"] = r.hash;
    return j;
}

json to_json(const Piece& piece) {
    json j = json::parse(programs::canonical_manifest(piece));
    if (!piece.hash.empty()) j["hash"] = piece.hash;
    json files = json::object();
    for (const char* part : {"common", "derive", "draw"})
        if (auto it = piece.files.find(std::string(part) + ".wgsl"); it != piece.files.end()) files[part] = it->second;
    if (!files.empty()) j["files"] = std::move(files);
    return j;
}

json to_json(const Program& p) {
    json j = json::object();
    j["id"] = p.id;
    if (p.title) j["title"] = *p.title;
    if (p.piece) {
        j["piece"] = to_json(*p.piece);
        if (p.panel) j["panel"] = *p.panel;
        if (!p.bind.empty()) j["bind"] = p.bind;
        if (!p.params.empty()) j["params"] = params_json(p.params);
        if (!p.x.is_null()) j["x"] = p.x;
        return j;
    }
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
    } else if (o.kind == OutputKind::tensor) {
        out["shape"] = o.shape;
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
    if (!p.params.empty()) j["params"] = params_json(p.params);
    if (!p.x.is_null()) j["x"] = p.x;
    return j;
}

json to_json(const Stage& s) {
    json j = json::object();
    j["panel"] = s.panel;
    j["entity"] = s.entity;
    j["kind"] = stage_kind_name(s.kind);
    j["wgsl"] = s.wgsl;
    if (!s.params.empty()) j["params"] = params_json(s.params);
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
    if (n.kind == LayoutNode::Kind::panel) {
        if (!n.min && n.x.is_null()) return n.panel;
        json leaf = json::object();  // the table { panel, min } of version 1
        leaf["panel"] = n.panel;
        if (n.min) leaf["min"] = json::array({(*n.min)[0], (*n.min)[1]});
        if (!n.x.is_null()) leaf["x"] = n.x;
        return leaf;
    }
    json j = json::object();
    json children = json::array();
    for (const LayoutNode& c : n.children) children.push_back(to_json(c));
    if (n.kind == LayoutNode::Kind::split) {
        j["split"] = to_string(n.split);
        if (n.ratio) j["ratio"] = *n.ratio;
        if (!n.shares.empty()) j["shares"] = n.shares;
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
    if (!spec.states.empty()) {
        json list = json::array();
        for (const StateVar& s : spec.states) list.push_back(to_json(s));
        j["state"] = std::move(list);
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
    if (!spec.pieces.empty()) {
        json list = json::array();
        for (const Piece& p : spec.pieces) list.push_back(to_json(p));
        j[kPiecesKey] = std::move(list);
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
