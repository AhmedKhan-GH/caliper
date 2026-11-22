#pragma once
// A test helper: the smallest value that a schema of the command layer's
// subset accepts, made generically, so tests can answer for any command of the
// table without knowing its schema. It fills required properties only, takes
// the first enum value, the minimum of a number, the shortest string and the
// shortest list. A pattern is not honoured; callers check the sample with
// validate_command_value() and pass over a command whose sample fails.
#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>

namespace caliper_builder::mcp::test {

inline nlohmann::json sample_for_schema(const nlohmann::json& s, int depth = 0) {
    using nlohmann::json;
    if (depth > 24 || !s.is_object()) return nullptr;
    const auto e = s.find("enum");
    if (e != s.end() && e->is_array() && !e->empty()) return (*e)[0];
    std::string type;
    const auto t = s.find("type");
    if (t != s.end() && t->is_string()) {
        type = t->get<std::string>();
    } else if (t != s.end() && t->is_array()) {
        for (const auto& x : *t)
            if (x.is_string() && x != "null") {
                type = x.get<std::string>();
                break;
            }
        if (type.empty()) return nullptr;
    }
    if (type == "object") {
        json out = json::object();
        const auto props = s.find("properties");
        const auto req = s.find("required");
        if (req != s.end() && req->is_array() && props != s.end() && props->is_object())
            for (const auto& r : *req)
                if (r.is_string() && props->contains(r.get<std::string>()))
                    out[r.get<std::string>()] = sample_for_schema((*props)[r.get<std::string>()], depth + 1);
        return out;
    }
    if (type == "array") {
        json out = json::array();
        const int n = s.contains("minItems") && s["minItems"].is_number_integer() ? s["minItems"].get<int>() : 0;
        for (int i = 0; i < std::min(n, 8); ++i)
            out.push_back(sample_for_schema(s.contains("items") ? s["items"] : json::object(), depth + 1));
        return out;
    }
    if (type == "string") {
        const int n = s.contains("minLength") && s["minLength"].is_number_integer() ? s["minLength"].get<int>() : 1;
        return std::string(std::size_t(std::max(1, std::min(n, 64))), 'a');
    }
    if (type == "integer") {
        if (s.contains("minimum") && s["minimum"].is_number()) return s["minimum"].get<long long>();
        return 0;
    }
    if (type == "number") {
        if (s.contains("minimum") && s["minimum"].is_number()) return s["minimum"].get<double>();
        return 0.0;
    }
    if (type == "boolean") return false;
    return nullptr;
}

}  // namespace caliper_builder::mcp::test
