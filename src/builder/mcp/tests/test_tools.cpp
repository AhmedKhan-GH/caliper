// The tools made from the command table, read generically: every command
// becomes one valid tool, and results and errors become CallToolResults.
#include <caliper_builder/mcp/tools.h>

#include "../codec.h"
#include "schema_sample.h"

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>

#include <doctest/doctest.h>

#include <set>

using namespace caliper_builder;
using namespace caliper_builder::mcp;
using nlohmann::json;

namespace {

CommandDescriptor png_descriptor() {
    CommandDescriptor d;
    d.name = "test.picture";
    d.summary = "Returns a picture.";
    d.args = {{"type", "object"}, {"properties", json::object()}, {"required", json::array()}, {"additionalProperties", false}};
    d.result = {{"type", "object"},
                {"properties",
                 {{"png_base64", {{"type", "string"}, {"description", "The PNG."}}},
                  {"width", {{"type", "integer"}, {"description", "Width."}}},
                  {"height", {{"type", "integer"}, {"description", "Height."}}}}},
                {"required", {"png_base64", "width", "height"}},
                {"additionalProperties", false}};
    d.examples = {json::object()};
    return d;
}

std::string tiny_png() {
    return std::string("\x89PNG\r\n\x1a\n", 8) + std::string("\0\0\0\x0d", 4) + "IHDR" + std::string("\0\0\0\x02", 4) +
           std::string("\0\0\0\x03", 4) + std::string(17, '\0');
}

json error_of(const json& tool_result) {
    REQUIRE(tool_result.value("isError", false));
    REQUIRE(tool_result["content"].size() == 1);
    const json e = json::parse(tool_result["content"][0]["text"].get<std::string>());
    return e["error"];
}

}  // namespace

TEST_CASE("every command of the table that agents may run becomes one valid tool") {
    // The person's own commands (agents false) are left out, unreported.
    std::vector<CommandDescriptor> table;
    for (const auto& d : command_table())
        if (d.agents) table.push_back(d);
    REQUIRE(table.size() == command_table().size() - 2);
    ToolTableBuild b = build_tool_table(command_table(), "built-in");
    REQUIRE(b.error.ok());
    REQUIRE(b.table);
    CHECK(b.dropped.empty());
    CHECK(b.table->commands.size() == table.size());
    CHECK(b.table->tools.size() == table.size());
    CHECK(b.table->source == "built-in");
    CHECK(b.table->fingerprint.size() == 64);

    std::set<std::string> names;
    for (std::size_t i = 0; i < table.size(); ++i) {
        const CommandDescriptor& d = table[i];
        const json& tool = b.table->tools[i];
        const std::string name = tool["name"];
        CAPTURE(name);
        CHECK(is_valid_mcp_tool_name(name));
        CHECK(names.insert(name).second);
        CHECK(command_name_from_tool(name) == d.name);
        CHECK(b.table->find_tool(name) == &b.table->commands[i]);
        CHECK(tool["title"] == d.name);
        CHECK(tool["description"] == d.summary);
        CHECK(tool["inputSchema"] == d.args);
        CHECK(tool["inputSchema"]["type"] == "object");
        // Annotations follow the flags; app.launch changes what the app shows.
        const json& ann = tool["annotations"];
        CHECK(ann["readOnlyHint"] == (!d.mutates && d.name != "app.launch"));
        CHECK(ann["destructiveHint"] == d.destructive);
        CHECK(ann["openWorldHint"] == false);
        // The output schema is the result schema, without a PNG's bytes.
        REQUIRE(tool.contains("outputSchema"));
        const json& out = tool["outputSchema"];
        CHECK(out["type"] == "object");
        CHECK(check_command_schema(out).ok());
        if (command_returns_png(d)) {
            CHECK_FALSE(out["properties"].contains("png_base64"));
            for (const auto& r : out.value("required", json::array())) CHECK(r != "png_base64");
        } else {
            CHECK(out == d.result);
        }
        for (const auto& ex : d.examples) CHECK(validate_command_value(tool["inputSchema"], ex).ok());
    }
}

TEST_CASE("tool names follow MCP's rule") {
    CHECK(is_valid_mcp_tool_name("panel_create"));
    CHECK(is_valid_mcp_tool_name("admin.tools-list_v2"));
    CHECK(is_valid_mcp_tool_name(std::string(128, 'a')));
    CHECK_FALSE(is_valid_mcp_tool_name(""));
    CHECK_FALSE(is_valid_mcp_tool_name(std::string(129, 'a')));
    CHECK_FALSE(is_valid_mcp_tool_name("panel create"));
    CHECK_FALSE(is_valid_mcp_tool_name("panel/create"));
    CHECK_FALSE(is_valid_mcp_tool_name("caf\xc3\xa9"));
}

TEST_CASE("the person's commands, undo and redo, are no tools for an agent") {
    ToolTableBuild built = build_tool_table(command_table(), "built-in");
    REQUIRE(built.table);
    ToolTableBuild app = parse_app_command_table(command_table_to_json());
    REQUIRE(app.table);
    for (const ToolTable* t : {built.table.get(), app.table.get()}) {
        CHECK(t->find_command("history.undo") == nullptr);
        CHECK(t->find_command("history.redo") == nullptr);
        CHECK(t->find_command("history.list") != nullptr);
        for (const auto& tool : t->tools) CHECK(tool["name"] != "history_undo");
    }
}

TEST_CASE("the app's table is parsed strictly and a bad command is dropped alone") {
    const json body = command_table_to_json();
    ToolTableBuild ok = parse_app_command_table(body);
    REQUIRE(ok.error.ok());
    CHECK(ok.table->source == "app");
    const std::size_t agents_may = command_table().size() - 2;  // not history.undo, history.redo
    CHECK(ok.table->commands.size() == agents_may);
    CHECK(ok.dropped.empty());
    // The same table from the app and built in has the same fingerprint.
    CHECK(ok.table->fingerprint == build_tool_table(command_table(), "built-in").table->fingerprint);
    json shown = body;
    json kept = json::array();
    for (const auto& c : body["commands"])
        if (c["agents"] == true) kept.push_back(c);
    shown["commands"] = kept;
    CHECK(ok.table->to_json() == shown);
    // An app that predates the flag: every command it sends is an agent's.
    json older = body;
    for (auto& c : older["commands"]) c.erase("agents");
    CHECK(parse_app_command_table(older).table->commands.size() == command_table().size());

    CHECK(parse_app_command_table(json::array()).error.code == "mcp.bad_command_table");
    json v1 = body;
    v1["version"] = 1;
    CHECK(parse_app_command_table(v1).error.code == "mcp.bad_command_table");
    json none = body;
    none.erase("commands");
    CHECK(parse_app_command_table(none).error.code == "mcp.bad_command_table");
    json many = {{"version", 0}, {"commands", json::array()}};
    for (std::size_t i = 0; i <= ToolTableLimits::max_commands; ++i) many["commands"].push_back(json::object());
    CHECK(parse_app_command_table(many).error.code == "mcp.bad_command_table");

    json mixed = body;
    REQUIRE(mixed["commands"].size() >= 3);
    mixed["commands"][0].erase("summary");                                        // a missing field
    mixed["commands"][1]["args"] = {{"type", "object"}, {"oneOf", json::array()}};  // outside the subset
    mixed["commands"].push_back(mixed["commands"][2]);                             // listed twice
    mixed["commands"].push_back(42);                                               // not an object
    ToolTableBuild part = parse_app_command_table(mixed);
    REQUIRE(part.error.ok());
    CHECK(part.dropped.size() == 4);
    CHECK(part.table->commands.size() == agents_may - 2);
    CHECK(part.table->find_command(command_table()[0].name) == nullptr);
    CHECK(part.table->find_command(command_table()[2].name) != nullptr);
}

TEST_CASE("a command's result becomes structured content and text") {
    for (const CommandDescriptor& d : command_table()) {
        if (command_returns_png(d)) continue;
        const json sample = test::sample_for_schema(d.result);
        if (!validate_command_value(d.result, sample, "result").ok()) continue;
        CAPTURE(d.name);
        const json r = tool_result_from_command(d, sample);
        CHECK(r["isError"] == false);
        CHECK(r["structuredContent"] == sample);
        REQUIRE(r["content"].size() == 1);
        CHECK(r["content"][0]["type"] == "text");
        CHECK(json::parse(r["content"][0]["text"].get<std::string>()) == sample);
    }
    // A result that breaks the schema is a tool error that still shows it.
    const CommandDescriptor& any = command_table().front();
    json wrong = test::sample_for_schema(any.result);
    wrong["not_a_field"] = "surprise";
    const json e = error_of(tool_result_from_command(any, wrong));
    CHECK(e["code"] == "mcp.invalid_result");
    CHECK(e["message"].get<std::string>().find("surprise") != std::string::npos);
    CHECK(error_of(tool_result_from_command(any, json::array()))["code"] == "mcp.invalid_result");
}

TEST_CASE("a PNG result becomes image content and its size stays structured") {
    const CommandDescriptor d = png_descriptor();
    const std::string b64 = detail::base64_encode(tiny_png());
    const json r = tool_result_from_command(d, {{"png_base64", b64}, {"width", 2}, {"height", 3}});
    REQUIRE(r["isError"] == false);
    REQUIRE(r["content"].size() == 2);
    CHECK(r["content"][0] == json{{"type", "image"}, {"data", b64}, {"mimeType", "image/png"}});
    CHECK(r["structuredContent"] == json{{"width", 2}, {"height", 3}});
    CHECK(json::parse(r["content"][1]["text"].get<std::string>()) == json{{"width", 2}, {"height", 3}});

    CHECK(error_of(tool_result_from_command(d, {{"png_base64", "bm90IGEgcG5n"}, {"width", 2}, {"height", 3}}))["code"] ==
          "mcp.invalid_result");
    CHECK(error_of(tool_result_from_command(d, {{"png_base64", "%%%"}, {"width", 2}, {"height", 3}}))["code"] ==
          "mcp.invalid_result");
    CHECK(error_of(tool_result_from_command(d, {{"width", 2}, {"height", 3}}))["code"] == "mcp.invalid_result");
}

TEST_CASE("an error becomes a tool error that carries its code") {
    const json r = tool_error_result({"command.denied", "The person denied it.", "args"}, "a-7");
    CHECK(r["isError"] == true);
    const json e = error_of(r);
    CHECK(e == json{{"code", "command.denied"}, {"message", "The person denied it."}, {"where", "args"}, {"approval", "a-7"}});
    CHECK_FALSE(error_of(tool_error_result({"x.y", "m", ""})).contains("approval"));
}
