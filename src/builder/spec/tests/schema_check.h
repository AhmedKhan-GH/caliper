#pragma once
// A test-only JSON Schema (draft 2020-12) checker for exactly the keywords that
// caliper-spec-v0.schema.json uses. It refuses a schema that uses any other
// keyword, so the schema cannot grow a rule that this checker silently skips.
//
// Implemented: boolean schemas, $ref (to a JSON pointer in the same document),
// type, const, enum, properties, additionalProperties, required, items,
// minItems, maxItems, uniqueItems, minLength, maxLength (in code points),
// pattern (ECMAScript, searched as JSON Schema says), minimum, maximum, anyOf,
// oneOf, if, then, else. Annotations ($schema, $id, $defs, title, description,
// $comment) are read and ignored. Integers are numbers with no fractional part.
#include <cmath>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace spec_test {

class SchemaChecker {
public:
    using json = nlohmann::json;

    explicit SchemaChecker(json root) : root_(std::move(root)) { supported(root_, "#"); }

    // The failures of `instance`, each with its instance path; empty when it passes.
    std::vector<std::string> errors(const json& instance) const {
        std::vector<std::string> out;
        check(root_, instance, "", &out, 0);
        return out;
    }

    bool passes(const json& instance) const { return check(root_, instance, "", nullptr, 0); }

    // The keywords the schema uses, for the test that lists them.
    const std::set<std::string>& keywords() const { return used_; }

private:
    static const std::set<std::string>& known() {
        static const std::set<std::string> k = {
            "$schema", "$id",       "$defs",      "title",    "description", "$comment", "$ref",
            "type",    "const",     "enum",       "properties", "additionalProperties", "required",
            "items",   "minItems",  "maxItems",   "uniqueItems", "minLength", "maxLength", "pattern",
            "minimum", "maximum",   "anyOf",      "oneOf",    "if",          "then",     "else"};
        return k;
    }

    // Walks the whole schema and throws on a keyword this checker does not know.
    void supported(const json& s, const std::string& where) {
        if (s.is_boolean()) return;
        if (!s.is_object()) throw std::runtime_error("schema at " + where + " is not an object or a boolean");
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (!known().count(it.key()))
                throw std::runtime_error("schema keyword " + it.key() + " at " + where + " is not implemented");
            used_.insert(it.key());
            const std::string k = it.key();
            const json& v = it.value();
            if (k == "$defs" || k == "properties") {
                for (auto d = v.begin(); d != v.end(); ++d) supported(d.value(), where + "/" + k + "/" + d.key());
            } else if (k == "anyOf" || k == "oneOf") {
                for (std::size_t i = 0; i < v.size(); ++i) supported(v[i], where + "/" + k + "/" + std::to_string(i));
            } else if (k == "items" || k == "additionalProperties" || k == "if" || k == "then" || k == "else") {
                supported(v, where + "/" + k);
            } else if (k == "$ref") {
                const std::string ref = v.get<std::string>();
                if (ref.rfind("#/", 0) != 0) throw std::runtime_error("only local $ref is implemented: " + ref);
            }
        }
    }

    const json& resolve(const std::string& ref) const {
        return root_.at(json::json_pointer(ref.substr(1)));
    }

    static bool type_is(const json& inst, const std::string& t) {
        if (t == "object") return inst.is_object();
        if (t == "array") return inst.is_array();
        if (t == "string") return inst.is_string();
        if (t == "boolean") return inst.is_boolean();
        if (t == "null") return inst.is_null();
        if (t == "number") return inst.is_number();
        if (t == "integer") {
            if (inst.is_number_integer()) return true;
            if (!inst.is_number_float()) return false;
            const double d = inst.get<double>();
            return std::isfinite(d) && std::floor(d) == d;
        }
        throw std::runtime_error("unknown type " + t);
    }

    static std::size_t code_points(const std::string& s) {
        std::size_t n = 0;
        for (char c : s)
            if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++n;
        return n;
    }

    static bool fail(std::vector<std::string>* out, const std::string& path, const std::string& what) {
        if (out) out->push_back((path.empty() ? std::string("(root)") : path) + ": " + what);
        return false;
    }

    bool check(const json& s, const json& inst, const std::string& path, std::vector<std::string>* out,
               int depth) const {
        if (depth > 200) throw std::runtime_error("the schema recursed too deeply at " + path);
        if (s.is_boolean()) return s.get<bool>() ? true : fail(out, path, "is not allowed here");
        bool ok = true;
        if (auto r = s.find("$ref"); r != s.end())
            ok = check(resolve(r->get<std::string>()), inst, path, out, depth + 1) && ok;
        if (auto t = s.find("type"); t != s.end()) {
            bool match = false;
            if (t->is_string()) {
                match = type_is(inst, t->get<std::string>());
            } else {
                for (const json& e : *t) match = match || type_is(inst, e.get<std::string>());
            }
            if (!match) return fail(out, path, "has the wrong type, not " + t->dump());
        }
        if (auto c = s.find("const"); c != s.end() && !(inst == *c)) ok = fail(out, path, "is not " + c->dump());
        if (auto e = s.find("enum"); e != s.end()) {
            bool any = false;
            for (const json& v : *e) any = any || inst == v;
            if (!any) ok = fail(out, path, "is not one of " + e->dump());
        }
        if (inst.is_object()) {
            const json* props = s.contains("properties") ? &s["properties"] : nullptr;
            if (auto req = s.find("required"); req != s.end())
                for (const json& k : *req)
                    if (!inst.contains(k.get<std::string>()))
                        ok = fail(out, path, "lacks the required key " + k.get<std::string>());
            for (auto it = inst.begin(); it != inst.end(); ++it) {
                const std::string child = path + "/" + it.key();
                if (props && props->contains(it.key())) {
                    ok = check((*props)[it.key()], it.value(), child, out, depth + 1) && ok;
                } else if (auto ap = s.find("additionalProperties"); ap != s.end()) {
                    ok = check(*ap, it.value(), child, out, depth + 1) && ok;
                }
            }
        }
        if (inst.is_array()) {
            if (auto items = s.find("items"); items != s.end())
                for (std::size_t i = 0; i < inst.size(); ++i)
                    ok = check(*items, inst[i], path + "/" + std::to_string(i), out, depth + 1) && ok;
            if (auto mn = s.find("minItems"); mn != s.end() && inst.size() < mn->get<std::size_t>())
                ok = fail(out, path, "has fewer than " + mn->dump() + " items");
            if (auto mx = s.find("maxItems"); mx != s.end() && inst.size() > mx->get<std::size_t>())
                ok = fail(out, path, "has more than " + mx->dump() + " items");
            if (auto u = s.find("uniqueItems"); u != s.end() && u->get<bool>())
                for (std::size_t i = 0; i < inst.size(); ++i)
                    for (std::size_t j = i + 1; j < inst.size(); ++j)
                        if (inst[i] == inst[j]) ok = fail(out, path, "repeats an item");
        }
        if (inst.is_string()) {
            const std::string& str = inst.get_ref<const std::string&>();
            if (auto mn = s.find("minLength"); mn != s.end() && code_points(str) < mn->get<std::size_t>())
                ok = fail(out, path, "is shorter than " + mn->dump());
            if (auto mx = s.find("maxLength"); mx != s.end() && code_points(str) > mx->get<std::size_t>())
                ok = fail(out, path, "is longer than " + mx->dump());
            if (auto p = s.find("pattern"); p != s.end()) {
                const std::regex re(p->get<std::string>(), std::regex::ECMAScript);
                if (!std::regex_search(str, re)) ok = fail(out, path, "does not match " + p->dump());
            }
        }
        if (inst.is_number()) {
            const double v = inst.get<double>();
            if (auto mn = s.find("minimum"); mn != s.end() && v < mn->get<double>())
                ok = fail(out, path, "is below " + mn->dump());
            if (auto mx = s.find("maximum"); mx != s.end() && v > mx->get<double>())
                ok = fail(out, path, "is above " + mx->dump());
        }
        if (auto any = s.find("anyOf"); any != s.end()) {
            bool one = false;
            for (const json& sub : *any) one = one || check(sub, inst, path, nullptr, depth + 1);
            if (!one) ok = fail(out, path, "matches none of anyOf");
        }
        if (auto of = s.find("oneOf"); of != s.end()) {
            int matches = 0;
            for (const json& sub : *of) matches += check(sub, inst, path, nullptr, depth + 1) ? 1 : 0;
            if (matches != 1) ok = fail(out, path, "matches " + std::to_string(matches) + " of oneOf, not exactly one");
        }
        if (auto cond = s.find("if"); cond != s.end()) {
            if (check(*cond, inst, path, nullptr, depth + 1)) {
                if (auto then = s.find("then"); then != s.end()) ok = check(*then, inst, path, out, depth + 1) && ok;
            } else {
                if (auto els = s.find("else"); els != s.end()) ok = check(*els, inst, path, out, depth + 1) && ok;
            }
        }
        return ok;
    }

    json root_;
    std::set<std::string> used_;
};

}  // namespace spec_test
