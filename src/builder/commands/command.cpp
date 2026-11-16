#include <caliper_builder/commands/command.h>
#include <caliper_builder/commands/schema.h>

#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

bool lower(char c) { return c >= 'a' && c <= 'z'; }
bool digit(char c) { return c >= '0' && c <= '9'; }

bool id_char(char c) {
    return lower(c) || (c >= 'A' && c <= 'Z') || digit(c) || c == '.' || c == '_' || c == ':' || c == '-';
}

bool key_char(char c) { return id_char(c) || c == '/'; }

Error invalid(std::string message, std::string where) {
    return Error("command.invalid_invocation", std::move(message), std::move(where));
}

// A string that is safe to dump: invalid UTF-8 is replaced.
json safe_string(const std::string& s) {
    json j = s;
    // Round-trip through a dump with replacement to drop invalid bytes.
    return json::parse(j.dump(-1, ' ', false, json::error_handler_t::replace), nullptr, false);
}

}  // namespace

const char* command_origin_name(CommandOrigin origin) {
    switch (origin) {
        case CommandOrigin::ui: return "ui";
        case CommandOrigin::builder: return "builder";
        case CommandOrigin::agent: return "agent";
        case CommandOrigin::mcp: return "mcp";
        case CommandOrigin::api: return "api";
        case CommandOrigin::viewer: return "viewer";
    }
    return "ui";
}

std::optional<CommandOrigin> command_origin_from_name(std::string_view name) {
    for (const CommandOrigin o : {CommandOrigin::ui, CommandOrigin::builder, CommandOrigin::agent,
                                  CommandOrigin::mcp, CommandOrigin::api, CommandOrigin::viewer})
        if (name == command_origin_name(o)) return o;
    return std::nullopt;
}

Error check_command_name(std::string_view name) {
    const Error bad("command.invalid_invocation",
                    "A command name is group.verb: a group of [a-z][a-z0-9]*, a dot and a verb of [a-z][a-z0-9_]*, "
                    "at most 64 characters.",
                    "name");
    if (name.empty() || name.size() > 64) return bad;
    const auto dot = name.find('.');
    if (dot == std::string_view::npos || dot == 0 || dot + 1 >= name.size()) return bad;
    const std::string_view group = name.substr(0, dot);
    const std::string_view verb = name.substr(dot + 1);
    if (!lower(group[0]) || !lower(verb[0])) return bad;
    for (const char c : group)
        if (!lower(c) && !digit(c)) return bad;
    for (const char c : verb)
        if (!lower(c) && !digit(c) && c != '_') return bad;
    return {};
}

Error check_command_descriptor(const CommandDescriptor& d) {
    auto bad = [&](std::string message, std::string where) {
        return Error("command.bad_descriptor", std::move(message), d.name + (where.empty() ? "" : "." + where));
    };
    if (Error err = check_command_name(d.name)) return Error("command.bad_descriptor", err.message, d.name);
    if (d.summary.empty()) return bad("A descriptor has a summary.", "summary");
    if (Error err = check_command_schema(d.args)) return bad(err.message, "args" + err.where.substr(6));
    if (!d.args.contains("type") || d.args["type"] != json("object"))
        return bad("The arguments' schema is an object schema.", "args");
    if (Error err = check_command_schema(d.result)) return bad(err.message, "result" + err.where.substr(6));
    for (std::size_t i = 0; i < d.examples.size(); ++i)
        if (Error err = validate_command_value(d.args, d.examples[i], "args"))
            return bad("Example " + std::to_string(i) + " fails the schema: " + err.message,
                       "examples[" + std::to_string(i) + "]");
    return {};
}

json command_descriptor_to_json(const CommandDescriptor& d) {
    return json{{"name", d.name},
                {"summary", d.summary},
                {"args", d.args},
                {"result", d.result},
                {"examples", d.examples},
                {"mutates", d.mutates},
                {"destructive", d.destructive},
                {"frame_thread", d.frame_thread},
                {"agents", d.agents}};
}

std::string command_tool_name(std::string_view command_name) {
    if (check_command_name(command_name)) return {};
    std::string out(command_name);
    out[out.find('.')] = '_';
    return out;
}

std::string command_name_from_tool(std::string_view tool_name) {
    const auto underscore = tool_name.find('_');
    if (underscore == std::string_view::npos) return {};
    std::string out(tool_name);
    out[underscore] = '.';
    if (check_command_name(out)) return {};
    return out;
}

Error check_command_invocation(const CommandInvocation& inv) {
    if (inv.id.size() > 128) return invalid("An invocation id has at most 128 characters.", "id");
    for (const char c : inv.id)
        if (!id_char(c)) return invalid("An invocation id holds only A-Z, a-z, 0-9, '.', '_', ':' and '-'.", "id");
    if (Error err = check_command_name(inv.name)) return err;
    if (!inv.args.is_object()) return invalid("The arguments are a JSON object.", "args");
    for (const auto* field : {&inv.group, &inv.merge_key, &inv.preview}) {
        const char* where = field == &inv.group ? "group" : field == &inv.merge_key ? "merge_key" : "preview";
        if (field->size() > 128) return invalid(std::string("A ") + where + " has at most 128 characters.", where);
        for (const char c : *field)
            if (!key_char(c))
                return invalid(std::string("A ") + where + " holds only A-Z, a-z, 0-9, '.', '_', ':', '/' and '-'.",
                               where);
    }
    return {};
}

json command_invocation_to_json(const CommandInvocation& inv) {
    return json{{"id", safe_string(inv.id)},
                {"name", safe_string(inv.name)},
                {"args", inv.args},
                {"origin", command_origin_name(inv.origin)},
                {"group", inv.group.empty() ? json(nullptr) : safe_string(inv.group)},
                {"merge_key", inv.merge_key.empty() ? json(nullptr) : safe_string(inv.merge_key)},
                {"preview", inv.preview.empty() ? json(nullptr) : safe_string(inv.preview)}};
}

CommandInvocationParse command_invocation_from_json(const json& v) {
    CommandInvocationParse out;
    if (!v.is_object()) {
        out.error = invalid("An invocation is a JSON object.", "");
        return out;
    }
    for (const auto& item : v.items()) {
        const std::string& k = item.key();
        if (k != "id" && k != "name" && k != "args" && k != "origin" && k != "group" && k != "merge_key" &&
            k != "preview") {
            out.error = invalid("An invocation has no field of this name.", k);
            return out;
        }
    }
    CommandInvocation& inv = out.invocation;
    if (v.contains("id") && !v["id"].is_null()) {
        if (!v["id"].is_string()) {
            out.error = invalid("The id is a string.", "id");
            return out;
        }
        inv.id = v["id"].get<std::string>();
    }
    if (!v.contains("name") || !v["name"].is_string()) {
        out.error = invalid("An invocation names its command.", "name");
        return out;
    }
    inv.name = v["name"].get<std::string>();
    if (v.contains("args") && !v["args"].is_null()) {
        if (!v["args"].is_object()) {
            out.error = invalid("The arguments are a JSON object.", "args");
            return out;
        }
        // Copying a JSON value recurses, so a hostile depth is refused first.
        if (Error err = measure_command_value(v["args"], "args")) {
            out.error = std::move(err);
            return out;
        }
        inv.args = v["args"];
    }
    if (!v.contains("origin") || !v["origin"].is_string()) {
        out.error = invalid("An invocation gives its origin.", "origin");
        return out;
    }
    const auto origin = command_origin_from_name(v["origin"].get_ref<const std::string&>());
    if (!origin) {
        out.error = invalid("The origin is ui, builder, agent, mcp, api or viewer.", "origin");
        return out;
    }
    inv.origin = *origin;
    for (const char* k : {"group", "merge_key", "preview"}) {
        if (!v.contains(k) || v[k].is_null()) continue;
        if (!v[k].is_string()) {
            out.error = invalid(std::string(k) + " is a string or null.", k);
            return out;
        }
        const std::string key = k;
        (key == "group" ? inv.group : key == "merge_key" ? inv.merge_key : inv.preview) = v[k].get<std::string>();
    }
    out.error = check_command_invocation(inv);
    return out;
}

CommandResult CommandResult::success(std::string id, json result) {
    CommandResult r;
    r.id = std::move(id);
    r.ok = true;
    r.result = std::move(result);
    return r;
}

CommandResult CommandResult::failure(std::string id, Error error) {
    CommandResult r;
    r.id = std::move(id);
    r.ok = false;
    r.error = std::move(error);
    if (r.error.code.empty()) r.error.code = "command.failed";
    return r;
}

json command_result_to_json(const CommandResult& r) {
    json out = json::object();
    out["id"] = safe_string(r.id);
    out["ok"] = r.ok;
    if (r.ok) {
        out["result"] = r.result;
    } else {
        json error = {{"code", safe_string(r.error.code)},
                      {"message", safe_string(r.error.message)},
                      {"where", safe_string(r.error.where)}};
        if (!r.approval.empty()) error["approval"] = safe_string(r.approval);
        out["error"] = std::move(error);
    }
    return out;
}

CommandResultParse command_result_from_json(const json& v) {
    CommandResultParse out;
    auto fail = [&](std::string message, std::string where) {
        out.error = Error("command.invalid_result", std::move(message), std::move(where));
        return out;
    };
    if (!v.is_object()) return fail("A result is a JSON object.", "");
    if (!v.contains("id") || !v["id"].is_string()) return fail("A result carries the invocation's id.", "id");
    if (!v.contains("ok") || !v["ok"].is_boolean()) return fail("A result says whether it is ok.", "ok");
    CommandResult& r = out.result;
    r.id = v["id"].get<std::string>();
    r.ok = v["ok"].get<bool>();
    for (const auto& item : v.items())
        if (item.key() != "id" && item.key() != "ok" && item.key() != (r.ok ? "result" : "error"))
            return fail("A result has no field of this name here.", item.key());
    if (r.ok) {
        if (!v.contains("result")) return fail("An ok result carries its result.", "result");
        r.result = v["result"];
        return out;
    }
    if (!v.contains("error") || !v["error"].is_object()) return fail("A failed result carries its error.", "error");
    const json& e = v["error"];
    for (const auto& item : e.items()) {
        const std::string& k = item.key();
        if (k != "code" && k != "message" && k != "where" && k != "approval")
            return fail("An error has no field of this name.", "error." + k);
        if (!item.value().is_string()) return fail("The fields of an error are strings.", "error." + k);
    }
    if (!e.contains("code") || e["code"].get<std::string>().empty()) return fail("An error has a code.", "error.code");
    r.error.code = e["code"].get<std::string>();
    if (e.contains("message")) r.error.message = e["message"].get<std::string>();
    if (e.contains("where")) r.error.where = e["where"].get<std::string>();
    if (e.contains("approval")) r.approval = e["approval"].get<std::string>();
    return out;
}

}  // namespace caliper_builder
