// The loader format v0: strict parsing from TOML or JSON, validation and the
// canonical TOML writer.
#include <caliper_builder/loaders/loader.h>

#include <toml++/toml.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

namespace caliper_builder {

using json = nlohmann::json;

namespace {

// Key paths ("output[0].mode") to the line and column where TOML wrote them.
using Positions = std::map<std::string, std::string>;

constexpr int kMaxDepth = 16;

std::string position_of(const toml::source_region& r) {
    return std::to_string(r.begin.line) + ":" + std::to_string(r.begin.column);
}

Error toml_to_json(const toml::node& node, const std::string& path, json* out, Positions* pos, int depth) {
    if (depth > kMaxDepth) return Error("loader.parse", "The recipe is nested too deeply.", position_of(node.source()));
    if (const toml::table* t = node.as_table()) {
        *out = json::object();
        for (auto&& [k, v] : *t) {
            const std::string key(k.str());
            const std::string child = path.empty() ? key : path + "." + key;
            (*pos)[child] = position_of(k.source());
            json value;
            if (Error e = toml_to_json(v, child, &value, pos, depth + 1)) return e;
            (*out)[key] = std::move(value);
        }
        return {};
    }
    if (const toml::array* a = node.as_array()) {
        *out = json::array();
        std::size_t i = 0;
        for (const toml::node& v : *a) {
            const std::string child = path + "[" + std::to_string(i++) + "]";
            (*pos)[child] = position_of(v.source());
            json value;
            if (Error e = toml_to_json(v, child, &value, pos, depth + 1)) return e;
            out->push_back(std::move(value));
        }
        return {};
    }
    if (node.is_string()) {
        *out = node.value<std::string>().value_or("");
    } else if (node.is_integer()) {
        *out = node.value<int64_t>().value_or(0);
    } else if (node.is_floating_point()) {
        *out = node.value<double>().value_or(0.0);
    } else if (node.is_boolean()) {
        *out = node.value<bool>().value_or(false);
    } else {
        return Error("loader.bad_value", "`" + path + "` is a date or time, which a loader does not use.",
                     position_of(node.source()));
    }
    return {};
}

// Builds errors whose `where` is a position when the text was TOML, or the key path.
struct Context {
    const Positions* positions = nullptr;

    std::string where(std::string path) const {
        if (!positions) return path;
        while (true) {
            auto it = positions->find(path);
            if (it != positions->end()) return it->second;
            const std::size_t cut = path.find_last_of(".[");
            if (cut == std::string::npos || cut == 0) return "1:1";
            path = path.substr(0, cut);
        }
    }
    Error error(const std::string& code, const std::string& message, const std::string& path) const {
        return Error(code, message, where(path));
    }
};

bool text_ok(const std::string& s, bool allow_newlines) {
    for (unsigned char c : s) {
        if (c == 0) return false;
        if (c < 0x20 && c != '\t' && !(allow_newlines && (c == '\n' || c == '\r'))) return false;
    }
    return true;
}

Error check_keys(const json& obj, const std::vector<std::string>& allowed, const std::string& path, const Context& ctx) {
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end()) {
            const std::string key = path.empty() ? it.key() : path + "." + it.key();
            return ctx.error("loader.unknown_key", "The key `" + key + "` is not part of the loader format.", key);
        }
    }
    return {};
}

Error get_string(const json& obj, const std::string& key, const std::string& path, const Context& ctx, bool required,
                 std::size_t max, bool newlines, std::optional<std::string>* out) {
    const std::string full = path + "." + key;
    auto it = obj.find(key);
    if (it == obj.end()) {
        if (required) return ctx.error("loader.missing_key", "The loader needs `" + full + "`.", path);
        return {};
    }
    if (!it->is_string()) return ctx.error("loader.bad_value", "`" + full + "` is a string.", full);
    const std::string s = it->get<std::string>();
    if (s.size() > max) return ctx.error("loader.too_large", "`" + full + "` is too long.", full);
    if (!text_ok(s, newlines)) return ctx.error("loader.bad_value", "`" + full + "` holds a control character.", full);
    *out = s;
    return {};
}

LoaderParse from_json(const json& j, const Context& ctx) {
    LoaderParse r;
    Loader& l = r.loader;
    if (!j.is_object()) {
        r.error = ctx.error("loader.parse", "A loader is a table of keys.", "");
        return r;
    }
    if ((r.error = check_keys(j, {"caliper_loader", "loader", "input", "reader", "output"}, "", ctx))) return r;
    auto version = j.find("caliper_loader");
    if (version == j.end()) {
        r.error = ctx.error("loader.missing_key", "The loader needs `caliper_loader = 0`, its format version.", "");
        return r;
    }
    if (!version->is_number_integer() || version->get<int64_t>() != 0) {
        r.error = ctx.error("loader.unsupported", "This Caliper reads loader format version 0 only.", "caliper_loader");
        return r;
    }
    std::optional<std::string> s;

    auto loader = j.find("loader");
    if (loader == j.end() || !loader->is_object()) {
        r.error = ctx.error("loader.missing_key", "The loader needs a [loader] table.", "loader");
        return r;
    }
    if ((r.error = check_keys(*loader, {"id", "title", "description"}, "loader", ctx))) return r;
    if ((r.error = get_string(*loader, "id", "loader", ctx, true, 64, false, &s))) return r;
    l.id = *s;
    if (!valid_id(l.id)) {
        r.error = ctx.error("loader.bad_id", "`loader.id` matches [a-z][a-z0-9_-]{0,63}.", "loader.id");
        return r;
    }
    s.reset();
    if ((r.error = get_string(*loader, "title", "loader", ctx, false, 256, false, &s))) return r;
    l.title = s.value_or("");
    s.reset();
    if ((r.error = get_string(*loader, "description", "loader", ctx, false, 4096, true, &s))) return r;
    l.description = s.value_or("");

    auto input = j.find("input");
    if (input == j.end() || !input->is_object()) {
        r.error = ctx.error("loader.missing_key", "The loader needs an [input] table.", "input");
        return r;
    }
    if ((r.error = check_keys(*input, {"path", "reader"}, "input", ctx))) return r;
    s.reset();
    if ((r.error = get_string(*input, "path", "input", ctx, true, 4096, false, &s))) return r;
    l.path = *s;
    if (l.path.empty()) {
        r.error = ctx.error("loader.bad_value", "`input.path` names a file or a glob.", "input.path");
        return r;
    }
    s.reset();
    if ((r.error = get_string(*input, "reader", "input", ctx, false, 64, false, &s))) return r;
    if (s) {
        ReaderKind k;
        if (!reader_from_name(*s, &k)) {
            std::string names;
            for (const std::string& n : reader_names()) names += (names.empty() ? "" : ", ") + n;
            r.error = ctx.error("loader.bad_value", "`input.reader` is one of " + names + ".", "input.reader");
            return r;
        }
        l.reader = k;
    }

    auto reader = j.find("reader");
    if (reader != j.end()) {
        if (!reader->is_object()) {
            r.error = ctx.error("loader.bad_value", "[reader] is a table of options.", "reader");
            return r;
        }
        for (auto it = reader->begin(); it != reader->end(); ++it) {
            if (it->is_object() || it->is_array()) {
                r.error = ctx.error("loader.bad_value", "`reader." + it.key() + "` is a single value.", "reader." + it.key());
                return r;
            }
            if (it->is_string() && !text_ok(it->get<std::string>(), false)) {
                r.error = ctx.error("loader.bad_value", "`reader." + it.key() + "` holds a control character.",
                                    "reader." + it.key());
                // A tab is a valid delimiter.
                if (it.key() == "delimiter" && it->get<std::string>() == "\t") r.error = {};
                if (r.error) return r;
            }
        }
        l.reader_options = *reader;
    }

    auto outputs = j.find("output");
    if (outputs == j.end() || !outputs->is_array() || outputs->empty()) {
        r.error = ctx.error("loader.missing_key", "The loader needs at least one [[output]].", "output");
        return r;
    }
    if (outputs->size() > kMaxLoaderOutputs) {
        r.error = ctx.error("loader.too_large", "A loader has at most 256 outputs.", "output");
        return r;
    }
    std::set<std::string> names;
    for (std::size_t i = 0; i < outputs->size(); ++i) {
        const json& o = (*outputs)[i];
        const std::string path = "output[" + std::to_string(i) + "]";
        if (!o.is_object()) {
            r.error = ctx.error("loader.bad_value", "`" + path + "` is a table.", path);
            return r;
        }
        if ((r.error = check_keys(o, {"name", "title", "mode", "sql", "tensor", "transform"}, path, ctx))) return r;
        LoaderOutput out;
        s.reset();
        if ((r.error = get_string(o, "name", path, ctx, true, 64, false, &s))) return r;
        out.name = *s;
        if (!valid_id(out.name)) {
            r.error = ctx.error("loader.bad_id", "`" + path + ".name` matches [a-z][a-z0-9_-]{0,63}.", path + ".name");
            return r;
        }
        if (!names.insert(out.name).second) {
            r.error = ctx.error("loader.name_collision", "Two outputs are named " + out.name + ".", path + ".name");
            return r;
        }
        s.reset();
        if ((r.error = get_string(o, "title", path, ctx, false, 256, false, &s))) return r;
        out.title = s.value_or("");
        s.reset();
        if ((r.error = get_string(o, "mode", path, ctx, false, 16, false, &s))) return r;
        if (s && *s != "view" && *s != "table") {
            r.error = ctx.error("loader.bad_value", "`" + path + ".mode` is \"view\" or \"table\".", path + ".mode");
            return r;
        }
        out.mode = s;
        s.reset();
        if ((r.error = get_string(o, "sql", path, ctx, false, kMaxLoaderSql, true, &s))) return r;
        out.sql = s;
        s.reset();
        if ((r.error = get_string(o, "tensor", path, ctx, false, 1024, false, &s))) return r;
        if (s && s->empty()) {
            r.error = ctx.error("loader.bad_value", "`" + path + ".tensor` names a tensor or is \"*\".", path + ".tensor");
            return r;
        }
        out.tensor = s;
        auto transforms = o.find("transform");
        if (transforms != o.end()) {
            if (!transforms->is_array()) {
                r.error = ctx.error("loader.bad_value", "`" + path + ".transform` is a list of tables.", path + ".transform");
                return r;
            }
            if (transforms->size() > kMaxLoaderTransforms) {
                r.error = ctx.error("loader.too_large", "An output has at most 64 transforms.", path + ".transform");
                return r;
            }
            for (std::size_t k = 0; k < transforms->size(); ++k) {
                const std::string tpath = path + ".transform[" + std::to_string(k) + "]";
                TensorTransform t;
                if (Error e = transform_from_json((*transforms)[k], &t)) {
                    const std::string field = e.where.empty() ? tpath : tpath + "." + e.where;
                    r.error = ctx.error(e.code, "`" + field + "`: " + e.message, field);
                    return r;
                }
                out.transforms.push_back(std::move(t));
            }
        }
        l.outputs.push_back(std::move(out));
    }
    if (std::optional<ReaderKind> k = declared_reader(l)) {
        if (Error e = check_loader_for_reader(l, *k)) {
            r.error = Error(e.code, e.message, ctx.where(e.where));
            return r;
        }
    }
    return r;
}

std::string toml_string(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20 || c == 0x7F) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04X", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

std::string toml_value(const json& v) {
    if (v.is_string()) return toml_string(v.get<std::string>());
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_number_integer()) return v.dump();
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (std::isnan(d)) return "nan";
        if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.17g", d);
        std::string out = buf;
        if (out.find_first_of(".eE") == std::string::npos) out += ".0";
        return out;
    }
    if (v.is_array()) {
        std::string out = "[";
        for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + toml_value(v[i]);
        return out + "]";
    }
    return "\"\"";
}

uint32_t fnv1a(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) {
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

}  // namespace

bool valid_id(const std::string& id) {
    if (id.empty() || id.size() > 64 || id[0] < 'a' || id[0] > 'z') return false;
    for (char c : id) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    }
    return true;
}

std::string make_id(const std::string& text, bool underscores_only) {
    std::string id;
    for (unsigned char c : text) {
        if (c >= 'A' && c <= 'Z') id += static_cast<char>(c - 'A' + 'a');
        else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') id += static_cast<char>(c);
        else if (c == '-' && !underscores_only) id += '-';
        else id += '_';
    }
    if (id.empty() || id[0] < 'a' || id[0] > 'z') id = "f_" + id;
    if (id.size() > 64) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%08x", fnv1a(text));
        id = id.substr(0, 55) + (underscores_only ? "_" : "-") + buf;
    }
    return id;
}

std::optional<ReaderKind> declared_reader(const Loader& l) {
    if (l.reader) return l.reader;
    ReaderKind k;
    if (reader_for_extension(l.path, &k)) return k;
    return std::nullopt;
}

Error check_loader_for_reader(const Loader& l, ReaderKind reader) {
    if (reader != ReaderKind::python) {
        if (Error e = validate_reader_options(reader, l.reader_options)) return e;
    }
    const std::string rname = reader_name(reader);
    for (std::size_t i = 0; i < l.outputs.size(); ++i) {
        const LoaderOutput& o = l.outputs[i];
        const std::string path = "output[" + std::to_string(i) + "]";
        if (reader == ReaderKind::python) continue;
        if (reader_is_tabular(reader)) {
            if (o.tensor) {
                return Error("loader.bad_value", "The " + rname + " reader makes tables, so `" + path +
                                                     ".tensor` does not apply.",
                             path + ".tensor");
            }
            if (!o.transforms.empty()) {
                return Error("loader.bad_value", "The " + rname + " reader makes tables, so `" + path +
                                                     ".transform` does not apply; use `sql`.",
                             path + ".transform");
            }
        } else {
            if (o.mode) {
                return Error("loader.bad_value", "The " + rname + " reader makes tensors, so `" + path +
                                                     ".mode` does not apply.",
                             path + ".mode");
            }
            if (o.sql) {
                return Error("loader.bad_value", "The " + rname + " reader makes tensors, so `" + path +
                                                     ".sql` does not apply; use `transform`.",
                             path + ".sql");
            }
            if (reader_is_single_tensor(reader) && o.tensor) {
                return Error("loader.bad_value", "A file of the " + rname + " reader holds one tensor, so leave out `" +
                                                     path + ".tensor`.",
                             path + ".tensor");
            }
            if (!reader_is_single_tensor(reader) && !o.tensor) {
                return Error("loader.missing_key", "The " + rname + " reader needs `" + path +
                                                       ".tensor`: a tensor's name in the file, or \"*\" for all.",
                             path);
            }
        }
    }
    return {};
}

LoaderParse parse_loader(const std::string& text) {
    LoaderParse r;
    if (text.size() > kMaxLoaderText) {
        r.error = Error("loader.too_large", "A loader recipe is at most 1 MiB of text.");
        return r;
    }
    toml::table table;
#if TOML_EXCEPTIONS
    try {
        table = toml::parse(text);
    } catch (const toml::parse_error& e) {
        r.error = Error("loader.parse", "The recipe is not valid TOML: " + std::string(e.description()) + ".",
                        position_of(e.source()));
        return r;
    } catch (...) {
        r.error = Error("loader.parse", "The recipe could not be parsed.");
        return r;
    }
#else
    toml::parse_result parsed = toml::parse(text);
    if (!parsed) {
        r.error = Error("loader.parse", "The recipe is not valid TOML: " + std::string(parsed.error().description()) + ".",
                        position_of(parsed.error().source()));
        return r;
    }
    table = std::move(parsed).table();
#endif
    json j;
    Positions positions;
    if ((r.error = toml_to_json(table, "", &j, &positions, 0))) return r;
    Context ctx;
    ctx.positions = &positions;
    return from_json(j, ctx);
}

LoaderParse loader_from_json(const json& value) {
    return from_json(value, Context{});
}

json Loader::to_json() const {
    json j;
    j["caliper_loader"] = 0;
    j["loader"] = {{"id", id}};
    if (!title.empty()) j["loader"]["title"] = title;
    if (!description.empty()) j["loader"]["description"] = description;
    j["input"] = {{"path", path}};
    if (reader) j["input"]["reader"] = reader_name(*reader);
    if (!reader_options.empty()) j["reader"] = reader_options;
    json outs = json::array();
    for (const LoaderOutput& o : outputs) {
        json oj = {{"name", o.name}};
        if (!o.title.empty()) oj["title"] = o.title;
        if (o.mode) oj["mode"] = *o.mode;
        if (o.sql) oj["sql"] = *o.sql;
        if (o.tensor) oj["tensor"] = *o.tensor;
        if (!o.transforms.empty()) {
            json ts = json::array();
            for (const TensorTransform& t : o.transforms) ts.push_back(transform_to_json(t));
            oj["transform"] = ts;
        }
        outs.push_back(std::move(oj));
    }
    j["output"] = outs;
    return j;
}

std::string write_loader(const Loader& l) {
    std::string out = "caliper_loader = 0\n\n[loader]\n";
    out += "id = " + toml_string(l.id) + "\n";
    if (!l.title.empty()) out += "title = " + toml_string(l.title) + "\n";
    if (!l.description.empty()) out += "description = " + toml_string(l.description) + "\n";
    out += "\n[input]\npath = " + toml_string(l.path) + "\n";
    if (l.reader) out += "reader = " + toml_string(reader_name(*l.reader)) + "\n";
    if (l.reader_options.is_object() && !l.reader_options.empty()) {
        out += "\n[reader]\n";
        for (auto it = l.reader_options.begin(); it != l.reader_options.end(); ++it) {
            out += it.key() + " = " + toml_value(it.value()) + "\n";
        }
    }
    static const char* kTransformKeys[] = {"axis", "index", "start", "stop", "step", "fn", "keepdims", "mode"};
    for (const LoaderOutput& o : l.outputs) {
        out += "\n[[output]]\nname = " + toml_string(o.name) + "\n";
        if (!o.title.empty()) out += "title = " + toml_string(o.title) + "\n";
        if (o.mode) out += "mode = " + toml_string(*o.mode) + "\n";
        if (o.sql) out += "sql = " + toml_string(*o.sql) + "\n";
        if (o.tensor) out += "tensor = " + toml_string(*o.tensor) + "\n";
        for (const TensorTransform& t : o.transforms) {
            const json tj = transform_to_json(t);
            out += "\n  [[output.transform]]\n  op = " + toml_value(tj["op"]) + "\n";
            for (const char* key : kTransformKeys) {
                if (tj.contains(key)) out += std::string("  ") + key + " = " + toml_value(tj[key]) + "\n";
            }
        }
    }
    return out;
}

}  // namespace caliper_builder
