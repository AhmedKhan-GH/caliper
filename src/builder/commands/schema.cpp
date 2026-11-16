#include <caliper_builder/commands/schema.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

constexpr const char* kTypes[] = {"object", "array", "string", "integer", "number", "boolean", "null"};
constexpr const char* kKeywords[] = {"type",      "properties", "required",  "additionalProperties",
                                     "items",     "minItems",   "maxItems",  "enum",
                                     "minimum",   "maximum",    "minLength", "maxLength",
                                     "pattern",   "description", "title",    "default"};

Error bad_schema(std::string message, std::string where) {
    return Error("command.bad_schema", std::move(message), std::move(where));
}

Error bad_value(std::string message, std::string where) {
    return Error("command.invalid_args", std::move(message), std::move(where));
}

bool is_type_name(const std::string& s) {
    return std::find_if(std::begin(kTypes), std::end(kTypes), [&](const char* t) { return s == t; }) !=
           std::end(kTypes);
}

bool is_integral(const json& v) {
    if (v.is_number_integer()) return true;
    if (!v.is_number_float()) return false;
    const double d = v.get<double>();
    return std::isfinite(d) && d == std::floor(d);
}

bool is_count(const json& v) { return is_integral(v) && v.get<double>() >= 0; }

bool has_type(const json& value, const std::string& type) {
    if (type == "object") return value.is_object();
    if (type == "array") return value.is_array();
    if (type == "string") return value.is_string();
    if (type == "integer") return is_integral(value);
    if (type == "number") return value.is_number();
    if (type == "boolean") return value.is_boolean();
    if (type == "null") return value.is_null();
    return false;
}

std::string type_words(const json& type) {
    auto article = [](const std::string& t) {
        if (t == "null") return std::string("null");
        if (t == "integer" || t == "object" || t == "array") return "an " + t;
        return "a " + t;
    };
    if (type.is_string()) return article(type.get<std::string>());
    std::string out;
    for (std::size_t i = 0; i < type.size(); ++i) {
        if (i > 0) out += i + 1 == type.size() ? " or " : ", ";
        out += article(type[i].get<std::string>());
    }
    return out;
}

std::string short_dump(const json& v) {
    std::string s = v.dump(-1, ' ', false, json::error_handler_t::replace);
    if (s.size() > 60) s = s.substr(0, 57) + "...";
    return s;
}

std::string number_text(double d) {
    if (d == std::floor(d) && std::fabs(d) < 1e15) return std::to_string(static_cast<long long>(d));
    std::string s = std::to_string(d);
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

// A small cache of compiled patterns. Schemas come from the table, so the set
// is small; the cache stops growing at 256 entries.
class PatternCache {
public:
    std::shared_ptr<const std::regex> get(const std::string& pattern, std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = cache_.find(pattern);
        if (it != cache_.end()) return it->second;
        std::shared_ptr<const std::regex> re;
        try {
            re = std::make_shared<const std::regex>(pattern, std::regex::ECMAScript);
        } catch (const std::exception& e) {
            error = e.what();
            return nullptr;
        }
        if (cache_.size() < 256) cache_.emplace(pattern, re);
        return re;
    }

private:
    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<const std::regex>> cache_;
};

PatternCache& patterns() {
    static PatternCache cache;
    return cache;
}

Error check_node(const json& s, const std::string& where, std::size_t depth) {
    if (depth > CommandSchemaLimits::max_schema_depth)
        return bad_schema("The schema nests deeper than 32 levels.", where);
    if (!s.is_object()) return bad_schema("A schema is a JSON object.", where);
    for (const auto& item : s.items()) {
        const std::string& k = item.key();
        if (std::find_if(std::begin(kKeywords), std::end(kKeywords), [&](const char* w) { return k == w; }) ==
            std::end(kKeywords))
            return bad_schema("The keyword " + k + " is not in the supported subset.", where);
    }
    if (s.contains("type")) {
        const json& t = s["type"];
        if (t.is_string()) {
            if (!is_type_name(t.get<std::string>())) return bad_schema("The type is not a JSON type.", where + ".type");
        } else if (t.is_array() && !t.empty()) {
            for (std::size_t i = 0; i < t.size(); ++i) {
                if (!t[i].is_string() || !is_type_name(t[i].get<std::string>()))
                    return bad_schema("The type list holds something that is not a JSON type.", where + ".type");
                for (std::size_t j = 0; j < i; ++j)
                    if (t[j] == t[i]) return bad_schema("The type list repeats a type.", where + ".type");
            }
        } else {
            return bad_schema("The type is a type name or a non-empty list of them.", where + ".type");
        }
    }
    if (s.contains("properties")) {
        if (!s["properties"].is_object()) return bad_schema("properties is an object.", where + ".properties");
        for (const auto& item : s["properties"].items())
            if (Error err = check_node(item.value(), where + ".properties." + item.key(), depth + 1)) return err;
    }
    if (s.contains("required")) {
        const json& r = s["required"];
        if (!r.is_array()) return bad_schema("required is a list of names.", where + ".required");
        for (std::size_t i = 0; i < r.size(); ++i) {
            if (!r[i].is_string()) return bad_schema("required is a list of names.", where + ".required");
            for (std::size_t j = 0; j < i; ++j)
                if (r[j] == r[i]) return bad_schema("required repeats a name.", where + ".required");
        }
    }
    if (s.contains("additionalProperties")) {
        const json& a = s["additionalProperties"];
        if (!a.is_boolean())
            if (Error err = check_node(a, where + ".additionalProperties", depth + 1)) return err;
    }
    if (s.contains("items"))
        if (Error err = check_node(s["items"], where + ".items", depth + 1)) return err;
    for (const char* k : {"minItems", "maxItems", "minLength", "maxLength"})
        if (s.contains(k) && !is_count(s[k]))
            return bad_schema(std::string(k) + " is a non-negative integer.", where + "." + k);
    for (const auto& pair : {std::pair<const char*, const char*>{"minItems", "maxItems"}, {"minLength", "maxLength"},
                             {"minimum", "maximum"}})
        if (s.contains(pair.first) && s.contains(pair.second) && s[pair.first].is_number() &&
            s[pair.second].is_number() && s[pair.first].get<double>() > s[pair.second].get<double>())
            return bad_schema(std::string(pair.first) + " is larger than " + pair.second + ".", where);
    for (const char* k : {"minimum", "maximum"})
        if (s.contains(k) && !s[k].is_number()) return bad_schema(std::string(k) + " is a number.", where + "." + k);
    if (s.contains("enum") && (!s["enum"].is_array() || s["enum"].empty()))
        return bad_schema("enum is a non-empty list.", where + ".enum");
    if (s.contains("pattern")) {
        if (!s["pattern"].is_string()) return bad_schema("pattern is a string.", where + ".pattern");
        std::string error;
        if (!patterns().get(s["pattern"].get<std::string>(), error))
            return bad_schema("The pattern does not compile: " + error, where + ".pattern");
    }
    for (const char* k : {"description", "title"})
        if (s.contains(k) && !s[k].is_string()) return bad_schema(std::string(k) + " is a string.", where + "." + k);
    return {};
}

Error validate_node(const json& s, const json& v, const std::string& where, std::size_t depth) {
    if (depth > CommandSchemaLimits::max_schema_depth)
        return Error("command.bad_schema", "The schema nests deeper than 32 levels.", where);
    if (s.is_boolean()) {
        if (s.get<bool>()) return {};
        return bad_value(where + " is not allowed.", where);
    }
    if (!s.is_object()) return {};
    if (s.contains("type")) {
        const json& t = s["type"];
        bool match = false;
        if (t.is_string()) {
            match = has_type(v, t.get<std::string>());
        } else if (t.is_array()) {
            for (const auto& one : t) match = match || (one.is_string() && has_type(v, one.get<std::string>()));
        }
        if (!match) return bad_value(where + " must be " + type_words(t) + ".", where);
    }
    if (s.contains("enum") && s["enum"].is_array()) {
        const json& e = s["enum"];
        if (std::find(e.begin(), e.end(), v) == e.end()) {
            std::string choices;
            for (std::size_t i = 0; i < e.size() && i < 12; ++i) {
                if (i > 0) choices += i + 1 == e.size() ? " or " : ", ";
                choices += short_dump(e[i]);
            }
            if (e.size() > 12) choices += " and others";
            return bad_value(where + " must be one of " + choices + ".", where);
        }
    }
    if (v.is_object()) {
        if (s.contains("required") && s["required"].is_array())
            for (const auto& name : s["required"])
                if (name.is_string() && !v.contains(name.get<std::string>()))
                    return bad_value(where + "." + name.get<std::string>() + " is required.",
                                     where + "." + name.get<std::string>());
        const json* props = s.contains("properties") && s["properties"].is_object() ? &s["properties"] : nullptr;
        const json* extra = s.contains("additionalProperties") ? &s["additionalProperties"] : nullptr;
        for (const auto& item : v.items()) {  // sorted by name
            const std::string at = where + "." + item.key();
            if (props && props->contains(item.key())) {
                if (Error err = validate_node((*props)[item.key()], item.value(), at, depth + 1)) return err;
            } else if (extra) {
                if (extra->is_boolean() && !extra->get<bool>())
                    return bad_value(where + " has no property named " + short_dump(json(item.key())) + ".", at);
                if (Error err = validate_node(*extra, item.value(), at, depth + 1)) return err;
            }
        }
    }
    if (v.is_array()) {
        const auto size = static_cast<double>(v.size());
        if (s.contains("minItems") && size < s["minItems"].get<double>())
            return bad_value(where + " must hold at least " + number_text(s["minItems"].get<double>()) + " items.",
                             where);
        if (s.contains("maxItems") && size > s["maxItems"].get<double>())
            return bad_value(where + " must hold at most " + number_text(s["maxItems"].get<double>()) + " items.",
                             where);
        if (s.contains("items"))
            for (std::size_t i = 0; i < v.size(); ++i)
                if (Error err = validate_node(s["items"], v[i], where + "[" + std::to_string(i) + "]", depth + 1))
                    return err;
    }
    if (v.is_string()) {
        const std::string& text = v.get_ref<const std::string&>();
        const auto length = static_cast<double>(command_code_points(text));
        if (s.contains("minLength") && length < s["minLength"].get<double>())
            return bad_value(where + " must be at least " + number_text(s["minLength"].get<double>()) +
                                 " characters long.",
                             where);
        if (s.contains("maxLength") && length > s["maxLength"].get<double>())
            return bad_value(where + " must be at most " + number_text(s["maxLength"].get<double>()) +
                                 " characters long.",
                             where);
        if (s.contains("pattern") && s["pattern"].is_string()) {
            if (text.size() > CommandSchemaLimits::max_pattern_input)
                return bad_value(where + " is too long to check against its pattern.", where);
            std::string error;
            const auto re = patterns().get(s["pattern"].get<std::string>(), error);
            bool matched = false;
            try {
                matched = re && std::regex_search(text, *re);
            } catch (const std::exception&) {
                return bad_value(where + " could not be checked against its pattern.", where);
            }
            if (!matched)
                return bad_value(where + " does not match the pattern " + s["pattern"].get<std::string>() + ".",
                                 where);
        }
    }
    if (v.is_number()) {
        const double d = v.get<double>();
        if (s.contains("minimum") && d < s["minimum"].get<double>())
            return bad_value(where + " must be at least " + number_text(s["minimum"].get<double>()) + ".", where);
        if (s.contains("maximum") && d > s["maximum"].get<double>())
            return bad_value(where + " must be at most " + number_text(s["maximum"].get<double>()) + ".", where);
    }
    return {};
}

bool is_object_schema(const json& s) {
    if (!s.is_object()) return false;
    if (s.contains("properties")) return true;
    if (!s.contains("type")) return false;
    const json& t = s["type"];
    if (t.is_string()) return t == "object";
    return std::find(t.begin(), t.end(), json("object")) != t.end();
}

Error check_closed_node(const json& s, const std::string& path, const std::vector<std::string>& open_paths,
                        std::size_t depth) {
    if (depth > CommandSchemaLimits::max_schema_depth) return bad_schema("The schema nests too deeply.", path);
    if (!s.is_object()) return bad_schema("A schema is a JSON object.", path);
    if (is_object_schema(s)) {
        const bool open = std::find(open_paths.begin(), open_paths.end(), path) != open_paths.end();
        if (!open) {
            if (!s.contains("additionalProperties") || s["additionalProperties"] != json(false))
                return bad_schema("An object schema must state additionalProperties: false.", path);
            if (!s.contains("required"))
                return bad_schema("An object schema must list its required properties, even when there are none.",
                                  path);
            if (!s.contains("properties")) return bad_schema("An object schema must list its properties.", path);
            if (!s["required"].is_array()) return bad_schema("required is a list of names.", path);
            for (const auto& name : s["required"])
                if (!name.is_string() || !s["properties"].contains(name.get<std::string>()))
                    return bad_schema("A required property is not among the properties.", path + "." +
                                                                                           name.get<std::string>());
        }
        if (s.contains("properties")) {
            for (const auto& item : s["properties"].items()) {
                const std::string at = path + "." + item.key();
                const json& p = item.value();
                if (!p.is_object() || !p.contains("description") || !p["description"].is_string() ||
                    p["description"].get<std::string>().empty())
                    return bad_schema("Every property must have a description.", at);
                if (Error err = check_closed_node(p, at, open_paths, depth + 1)) return err;
            }
        }
    }
    if (s.contains("items"))
        if (Error err = check_closed_node(s["items"], path + "[]", open_paths, depth + 1)) return err;
    if (s.contains("additionalProperties") && s["additionalProperties"].is_object())
        return bad_schema("additionalProperties must be false, not a schema.", path);
    return {};
}

}  // namespace

std::size_t command_code_points(std::string_view text) {
    std::size_t n = 0;
    for (const char c : text)
        if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++n;
    return n;
}

Error check_command_schema(const json& schema) {
    try {
        return check_node(schema, "schema", 0);
    } catch (const std::exception& e) {
        return bad_schema(std::string("The schema could not be checked: ") + e.what(), "schema");
    }
}

Error measure_command_value(const json& value, std::string_view root) {
    // An explicit stack, so a value nested a million levels deep cannot
    // overflow the call stack.
    std::vector<std::pair<const json*, std::size_t>> stack;
    stack.emplace_back(&value, 0);
    std::size_t nodes = 0;
    while (!stack.empty()) {
        const auto [node, depth] = stack.back();
        stack.pop_back();
        if (++nodes > CommandSchemaLimits::max_value_nodes)
            return bad_value(std::string(root) + " holds more than 1,048,576 values.", std::string(root));
        if (node->is_structured()) {
            if (depth + 1 > CommandSchemaLimits::max_value_depth)
                return bad_value(std::string(root) + " is nested more than 64 levels deep.", std::string(root));
            for (const auto& child : *node) stack.emplace_back(&child, depth + 1);
        }
    }
    return {};
}

Error validate_command_value(const json& schema, const json& value, std::string_view root) {
    if (Error err = measure_command_value(value, root)) return err;
    try {
        return validate_node(schema, value, std::string(root), 0);
    } catch (const std::exception& e) {
        // Only a schema that check_command_schema() would refuse gets here.
        return bad_schema(std::string("The schema could not be applied: ") + e.what(), std::string(root));
    }
}

Error check_closed_command_schema(const json& schema, std::string_view root,
                                  const std::vector<std::string>& open_paths) {
    try {
        return check_closed_node(schema, std::string(root), open_paths, 0);
    } catch (const std::exception& e) {
        return bad_schema(std::string("The schema could not be checked: ") + e.what(), std::string(root));
    }
}

}  // namespace caliper_builder
