// MCP tools from the command table (tools.h).
#include <caliper_builder/mcp/tools.h>

#include "codec.h"

#include <caliper_builder/commands/schema.h>
#include <caliper_builder/commands/sha256.h>

#include <set>

namespace caliper_builder::mcp {

using nlohmann::json;

namespace {

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// Commands that do not mutate the workspace but change what the app shows,
// so they are not read-only in MCP's sense.
bool changes_the_app_view(const CommandDescriptor& d) { return d.name == "app.launch"; }

}  // namespace

bool is_valid_mcp_tool_name(std::string_view name) {
    if (name.empty() || name.size() > 128) return false;
    for (char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                        c == '-' || c == '.';
        if (!ok) return false;
    }
    return true;
}

bool command_returns_png(const CommandDescriptor& d) {
    const auto props = d.result.find("properties");
    return props != d.result.end() && props->is_object() && props->contains("png_base64");
}

json tool_output_schema(const CommandDescriptor& d) {
    const auto type = d.result.find("type");
    if (!d.result.is_object() || type == d.result.end() || *type != "object") return nullptr;
    json schema = d.result;
    if (!command_returns_png(d)) return schema;
    schema["properties"].erase("png_base64");
    auto req = schema.find("required");
    if (req != schema.end() && req->is_array()) {
        json kept = json::array();
        for (const auto& r : *req)
            if (r != "png_base64") kept.push_back(r);
        *req = std::move(kept);
    }
    return schema;
}

json tool_definition(const CommandDescriptor& d) {
    const bool read_only = !d.mutates && !changes_the_app_view(d);
    json tool = {
        {"name", command_tool_name(d.name)},
        {"title", d.name},
        {"description", d.summary},
        {"inputSchema", d.args},
        {"annotations",
         {{"readOnlyHint", read_only}, {"destructiveHint", d.destructive}, {"openWorldHint", false}}},
    };
    json out = tool_output_schema(d);
    if (!out.is_null()) tool["outputSchema"] = std::move(out);
    return tool;
}

const CommandDescriptor* ToolTable::find_tool(std::string_view tool_name) const {
    for (std::size_t i = 0; i < tool_names.size(); ++i)
        if (tool_names[i] == tool_name) return &commands[i];
    return nullptr;
}

const CommandDescriptor* ToolTable::find_command(std::string_view command_name) const {
    for (const auto& d : commands)
        if (d.name == command_name) return &d;
    return nullptr;
}

json ToolTable::to_json() const {
    json list = json::array();
    for (const auto& d : commands) list.push_back(command_descriptor_to_json(d));
    return {{"version", 0}, {"commands", std::move(list)}};
}

ToolTableBuild build_tool_table(std::vector<CommandDescriptor> descriptors, std::string source) {
    ToolTableBuild out;
    if (descriptors.size() > ToolTableLimits::max_commands) {
        out.error = {"mcp.bad_command_table",
                     "The command table has more than " + std::to_string(ToolTableLimits::max_commands) + " commands.",
                     source};
        return out;
    }
    auto table = std::make_shared<ToolTable>();
    table->source = std::move(source);
    std::set<std::string> names;
    for (auto& d : descriptors) {
        // The person's own commands (history.undo, history.redo) are no tools
        // for an agent; the app refuses them to the agent and mcp origins.
        if (!d.agents) continue;
        if (Error err = check_command_descriptor(d)) {
            out.dropped.push_back({"mcp.bad_descriptor", err.message, d.name.size() <= 64 ? d.name : std::string()});
            continue;
        }
        const std::string tool = command_tool_name(d.name);
        if (!is_valid_mcp_tool_name(tool)) {
            out.dropped.push_back({"mcp.bad_descriptor", "The command's tool name breaks MCP's name rule.", d.name});
            continue;
        }
        if (!names.insert(tool).second) {
            out.dropped.push_back({"mcp.bad_descriptor", "The command is listed twice.", d.name});
            continue;
        }
        table->tools.push_back(tool_definition(d));
        table->tool_names.push_back(tool);
        table->commands.push_back(std::move(d));
    }
    table->fingerprint = command_sha256_hex(dump(table->to_json()));
    out.table = std::move(table);
    return out;
}

ToolTableBuild parse_app_command_table(const json& body) {
    ToolTableBuild out;
    auto refuse = [&](const std::string& why) {
        out.error = {"mcp.bad_command_table", "The app's command table is malformed: " + why + ".", "GET /v0/commands"};
        return out;
    };
    if (!body.is_object()) return refuse("it is not a JSON object");
    const auto version = body.find("version");
    if (version == body.end() || !version->is_number_integer() || version->get<std::int64_t>() != 0)
        return refuse("its version is not 0");
    const auto commands = body.find("commands");
    if (commands == body.end() || !commands->is_array()) return refuse("it has no list of commands");
    if (commands->size() > ToolTableLimits::max_commands)
        return refuse("it has more than " + std::to_string(ToolTableLimits::max_commands) + " commands");
    std::vector<CommandDescriptor> descriptors;
    std::vector<Error> dropped;
    std::size_t index = 0;
    for (const auto& c : *commands) {
        const std::string where = "commands[" + std::to_string(index++) + "]";
        auto field = [&](const char* name, json::value_t type) -> const json* {
            if (!c.is_object()) return nullptr;
            const auto it = c.find(name);
            if (it == c.end()) return nullptr;
            if (type == json::value_t::boolean ? !it->is_boolean()
                : type == json::value_t::string ? !it->is_string()
                : type == json::value_t::array  ? !it->is_array()
                                                : !it->is_object())
                return nullptr;
            return &*it;
        };
        const json* name = field("name", json::value_t::string);
        const json* summary = field("summary", json::value_t::string);
        const json* args = field("args", json::value_t::object);
        const json* result = field("result", json::value_t::object);
        const json* examples = field("examples", json::value_t::array);
        const json* mutates = field("mutates", json::value_t::boolean);
        const json* destructive = field("destructive", json::value_t::boolean);
        const json* frame_thread = field("frame_thread", json::value_t::boolean);
        if (!name || !summary || !args || !result || !examples || !mutates || !destructive || !frame_thread) {
            dropped.push_back({"mcp.bad_descriptor", "The command lacks a descriptor field or has one of the wrong type.", where});
            continue;
        }
        CommandDescriptor d;
        d.name = name->get<std::string>();
        d.summary = summary->get<std::string>();
        d.args = *args;
        d.result = *result;
        for (const auto& e : *examples) d.examples.push_back(e);
        d.mutates = mutates->get<bool>();
        d.destructive = destructive->get<bool>();
        d.frame_thread = frame_thread->get<bool>();
        // An app that predates the flag sends none; its commands are agents'.
        const auto agents = c.find("agents");
        if (agents != c.end() && agents->is_boolean()) d.agents = agents->get<bool>();
        descriptors.push_back(std::move(d));
    }
    out = build_tool_table(std::move(descriptors), "app");
    out.dropped.insert(out.dropped.begin(), dropped.begin(), dropped.end());
    return out;
}

json tool_error_result(const Error& error, const std::string& approval) {
    json e = {{"code", error.code}, {"message", error.message}, {"where", error.where}};
    if (!approval.empty()) e["approval"] = approval;
    return {{"content", json::array({{{"type", "text"}, {"text", dump(json{{"error", std::move(e)}})}}})},
            {"isError", true}};
}

json tool_result_from_command(const CommandDescriptor& d, const json& result) {
    if (!result.is_object()) {
        return tool_error_result({"mcp.invalid_result", "The app answered with a result that is not a JSON object: " +
                                                            dump(result).substr(0, 4096),
                                  "result"});
    }
    json structured = result;
    json content = json::array();
    if (command_returns_png(d)) {
        const auto png = structured.find("png_base64");
        std::string bytes;
        std::uint32_t w = 0, h = 0;
        if (png == structured.end() || !png->is_string() || png->get_ref<const std::string&>().size() % 4 != 0 ||
            !detail::base64_decode(png->get_ref<const std::string&>().substr(0, 64), &bytes) ||
            !detail::png_size(bytes, &w, &h)) {
            structured.erase("png_base64");
            return tool_error_result({"mcp.invalid_result",
                                      "The app's result has no PNG in png_base64. The rest of the result: " +
                                          dump(structured).substr(0, 4096),
                                      "result.png_base64"});
        }
        content.push_back({{"type", "image"}, {"data", png->get<std::string>()}, {"mimeType", "image/png"}});
        structured.erase("png_base64");
    }
    const json schema = tool_output_schema(d);
    if (!schema.is_null()) {
        if (Error err = validate_command_value(schema, structured, "result")) {
            return tool_error_result({"mcp.invalid_result",
                                      "The app's result does not match the command's result schema (" + err.message +
                                          "). The result: " + dump(structured).substr(0, 65536),
                                      err.where});
        }
    }
    content.push_back({{"type", "text"}, {"text", dump(structured)}});
    return {{"content", std::move(content)}, {"structuredContent", std::move(structured)}, {"isError", false}};
}

}  // namespace caliper_builder::mcp
