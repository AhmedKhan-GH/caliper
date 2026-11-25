// Tests of the built-in schema validator: each keyword, the paths of faults,
// the subset check, the closed-schema rules and hostile values.
#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/schema.h>

#include <doctest/doctest.h>

#include <string>

using nlohmann::json;
namespace cb = caliper_builder;

namespace {

// The `where` of the first fault, or "ok".
std::string fault(const json& schema, const json& value) {
    REQUIRE(cb::check_command_schema(schema).ok());
    const auto err = cb::validate_command_value(schema, value);
    if (!err) return "ok";
    CHECK(err.code == "command.invalid_args");
    CHECK(!err.message.empty());
    return err.where;
}

std::string schema_fault(const json& schema) {
    const auto err = cb::check_command_schema(schema);
    if (!err) return "ok";
    CHECK(err.code == "command.bad_schema");
    return err.where;
}

}  // namespace

TEST_CASE("type accepts one name or a list, and integer accepts integral numbers") {
    CHECK(fault({{"type", "string"}}, "x") == "ok");
    CHECK(fault({{"type", "string"}}, 1) == "args");
    CHECK(fault({{"type", "integer"}}, 2) == "ok");
    CHECK(fault({{"type", "integer"}}, 2.0) == "ok");
    CHECK(fault({{"type", "integer"}}, 2.5) == "args");
    CHECK(fault({{"type", "number"}}, 2.5) == "ok");
    CHECK(fault({{"type", "number"}}, "2.5") == "args");
    CHECK(fault({{"type", "boolean"}}, false) == "ok");
    CHECK(fault({{"type", "null"}}, nullptr) == "ok");
    CHECK(fault({{"type", "array"}}, json::array()) == "ok");
    CHECK(fault({{"type", "object"}}, json::object()) == "ok");
    CHECK(fault({{"type", "object"}}, json::array()) == "args");
    CHECK(fault({{"type", {"string", "null"}}}, nullptr) == "ok");
    CHECK(fault({{"type", {"string", "null"}}}, 3) == "args");
    const auto err = cb::validate_command_value({{"type", {"string", "integer", "null"}}}, true);
    CHECK(err.message == "args must be a string, an integer or null.");
}

TEST_CASE("properties, required and additionalProperties") {
    const json s = json::parse(R"({
      "type": "object",
      "properties": {"id": {"type": "string"}, "limit": {"type": "integer"}},
      "required": ["id"],
      "additionalProperties": false})");
    CHECK(fault(s, {{"id", "a"}}) == "ok");
    CHECK(fault(s, {{"id", "a"}, {"limit", 3}}) == "ok");
    CHECK(fault(s, {{"limit", 3}}) == "args.id");
    CHECK(fault(s, {{"id", 1}}) == "args.id");
    CHECK(fault(s, {{"id", "a"}, {"extra", 1}}) == "args.extra");
    const auto err = cb::validate_command_value(s, {{"id", "a"}, {"extra", 1}});
    CHECK(err.message == "args has no property named \"extra\".");

    json open = s;
    open.erase("additionalProperties");
    CHECK(fault(open, {{"id", "a"}, {"extra", 1}}) == "ok");
    open["additionalProperties"] = true;
    CHECK(fault(open, {{"id", "a"}, {"extra", 1}}) == "ok");
    open["additionalProperties"] = {{"type", "string"}};
    CHECK(fault(open, {{"id", "a"}, {"extra", 1}}) == "args.extra");
    CHECK(fault(open, {{"id", "a"}, {"extra", "x"}}) == "ok");
}

TEST_CASE("the first fault is deterministic: required first, then properties by name") {
    const json s = json::parse(R"({
      "type": "object",
      "properties": {"b": {"type": "string"}, "a": {"type": "string"}, "c": {"type": "string"}},
      "required": ["c"]})");
    CHECK(fault(s, {{"b", 1}, {"a", 1}}) == "args.c");
    CHECK(fault(s, {{"b", 1}, {"a", 1}, {"c", "x"}}) == "args.a");
}

TEST_CASE("nested paths name the value exactly") {
    const json s = json::parse(R"({
      "type": "object",
      "properties": {"bind": {"type": "object", "properties": {
          "y": {"type": "array", "items": {"type": "string", "minLength": 1}}}}}})");
    CHECK(fault(s, {{"bind", {{"y", {"a", ""}}}}}) == "args.bind.y[1]");
}

TEST_CASE("items, minItems and maxItems") {
    const json s = {{"type", "array"}, {"items", {{"type", "integer"}}}, {"minItems", 1}, {"maxItems", 3}};
    CHECK(fault(s, {1, 2, 3}) == "ok");
    CHECK(fault(s, json::array()) == "args");
    CHECK(fault(s, {1, 2, 3, 4}) == "args");
    CHECK(fault(s, {1, "x"}) == "args[1]");
    CHECK(cb::validate_command_value(s, {1, 2, 3, 4}).message == "args must hold at most 3 items.");
}

TEST_CASE("enum compares JSON values, numbers by value") {
    const json s = {{"enum", {"table", "tensor", 3, nullptr}}};
    CHECK(fault(s, "table") == "ok");
    CHECK(fault(s, 3.0) == "ok");
    CHECK(fault(s, nullptr) == "ok");
    CHECK(fault(s, "Table") == "args");
    CHECK(cb::validate_command_value(s, "x").message == "args must be one of \"table\", \"tensor\", 3 or null.");
}

TEST_CASE("minimum and maximum are inclusive") {
    const json s = {{"type", "number"}, {"minimum", 0.05}, {"maximum", 0.95}};
    CHECK(fault(s, 0.05) == "ok");
    CHECK(fault(s, 0.95) == "ok");
    CHECK(fault(s, 0.04) == "args");
    CHECK(fault(s, 1) == "args");
    const json i = {{"type", "integer"}, {"minimum", 1}, {"maximum", 1000}};
    CHECK(cb::validate_command_value(i, 0).message == "args must be at least 1.");
    CHECK(cb::validate_command_value(i, 1001).message == "args must be at most 1000.");
}

TEST_CASE("minLength and maxLength count code points") {
    const json s = {{"type", "string"}, {"minLength", 1}, {"maxLength", 3}};
    CHECK(fault(s, "abc") == "ok");
    CHECK(fault(s, "") == "args");
    CHECK(fault(s, "abcd") == "args");
    CHECK(fault(s, "\xC3\xA9\xC3\xA9\xC3\xA9") == "ok");      // three e-acute, six bytes
    CHECK(fault(s, "\xF0\x9F\x98\x80\xF0\x9F\x98\x80") == "ok");  // two four-byte characters
    CHECK(cb::command_code_points("\xF0\x9F\x98\x80x") == 2);
}

TEST_CASE("pattern searches the string, and only up to 4096 bytes") {
    const json s = {{"type", "string"}, {"pattern", "^[a-z][a-z0-9_-]{0,63}$"}};
    CHECK(fault(s, "heat") == "ok");
    CHECK(fault(s, "Heat") == "args");
    CHECK(fault(s, std::string(65, 'a')) == "args");
    const json anywhere = {{"type", "string"}, {"pattern", "b"}};
    CHECK(fault(anywhere, "abc") == "ok");
    const json loose = {{"type", "string"}, {"pattern", "a*"}};
    CHECK(fault(loose, std::string(4096, 'a')) == "ok");
    const auto err = cb::validate_command_value(loose, std::string(4097, 'a'));
    CHECK(err.message == "args is too long to check against its pattern.");
}

TEST_CASE("annotations are not checked") {
    const json s = {{"type", "string"}, {"description", "An id."}, {"title", "Id"}, {"default", 3}};
    CHECK(fault(s, "x") == "ok");
}

TEST_CASE("the subset check refuses everything outside the subset") {
    CHECK(schema_fault({{"type", "string"}}) == "ok");
    CHECK(schema_fault({{"oneOf", json::array()}}) == "schema");
    CHECK(schema_fault({{"$ref", "#/x"}}) == "schema");
    CHECK(schema_fault({{"format", "date-time"}}) == "schema");
    CHECK(schema_fault(json::array()) == "schema");
    CHECK(schema_fault({{"type", "text"}}) == "schema.type");
    CHECK(schema_fault({{"type", json::array()}}) == "schema.type");
    CHECK(schema_fault({{"type", {"string", "string"}}}) == "schema.type");
    CHECK(schema_fault({{"required", {"a", "a"}}}) == "schema.required");
    CHECK(schema_fault({{"required", "a"}}) == "schema.required");
    CHECK(schema_fault({{"minLength", -1}}) == "schema.minLength");
    CHECK(schema_fault({{"maxItems", 1.5}}) == "schema.maxItems");
    CHECK(schema_fault({{"minItems", 3}, {"maxItems", 2}}) == "schema");
    CHECK(schema_fault({{"minimum", "0"}}) == "schema.minimum");
    CHECK(schema_fault({{"enum", json::array()}}) == "schema.enum");
    CHECK(schema_fault({{"pattern", "(unclosed"}}) == "schema.pattern");
    CHECK(schema_fault({{"description", 3}}) == "schema.description");
    CHECK(schema_fault({{"properties", {{"a", {{"type", "nope"}}}}}}) == "schema.properties.a.type");
    CHECK(schema_fault({{"items", {{"bad", 1}}}}) == "schema.items");
    CHECK(schema_fault({{"additionalProperties", {{"type", "string"}}}}) == "ok");

    json deep = {{"type", "string"}};
    for (int i = 0; i < 32; ++i) deep = {{"type", "array"}, {"items", deep}};
    CHECK(schema_fault(deep) == "ok");
    deep = {{"type", "array"}, {"items", deep}};
    CHECK(cb::check_command_schema(deep).code == "command.bad_schema");
}

TEST_CASE("closed schemas state additionalProperties false, list required and describe every property") {
    const json good = json::parse(R"({
      "type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"type": "string", "description": "The id."},
        "place": {"type": ["object", "null"], "description": "Where.", "additionalProperties": false,
                  "required": [], "properties": {"ratio": {"type": "number", "description": "A share."}}},
        "list": {"type": "array", "description": "Items.", "items": {"type": "object",
                 "additionalProperties": false, "required": [], "properties": {}}}}})");
    CHECK(cb::check_closed_command_schema(good, "args").ok());

    json s = good;
    s.erase("additionalProperties");
    CHECK(cb::check_closed_command_schema(s, "args").where == "args");
    s = good;
    s.erase("required");
    CHECK(cb::check_closed_command_schema(s, "args").where == "args");
    s = good;
    s["properties"]["id"].erase("description");
    CHECK(cb::check_closed_command_schema(s, "args").where == "args.id");
    s = good;
    s["properties"]["place"]["additionalProperties"] = true;
    CHECK(cb::check_closed_command_schema(s, "args").where == "args.place");
    CHECK(cb::check_closed_command_schema(s, "args", {"args.place"}).ok());
    s = good;
    s["properties"]["list"]["items"].erase("required");
    CHECK(cb::check_closed_command_schema(s, "args").where == "args.list[]");
    s = good;
    s["required"] = {"missing"};
    CHECK(cb::check_closed_command_schema(s, "args").where == "args.missing");
}

TEST_CASE("closed schemas permit typed maps only at explicit open paths") {
    json schema = {{"type", "object"}, {"additionalProperties", {{"type", "integer"}}}};
    CHECK(cb::check_command_schema(schema).ok());
    CHECK(cb::check_closed_command_schema(schema, "result.steps").where == "result.steps");
    CHECK(cb::check_closed_command_schema(schema, "result.steps", {"result.other"}).where ==
          "result.steps");
    CHECK(cb::check_closed_command_schema(schema, "result.steps", {"result.steps"}).ok());
    CHECK(cb::validate_command_value(schema, {{"anchors", 3}, {"labels", 2}}).ok());
    CHECK_FALSE(cb::validate_command_value(schema, {{"anchors", "3"}}).ok());

    schema["additionalProperties"] = {{"type", "object"}, {"additionalProperties", true}};
    CHECK(cb::check_closed_command_schema(schema, "result.steps", {"result.steps"}).where ==
          "result.steps{}");
    schema["additionalProperties"] = {{"type", "object"}, {"additionalProperties", false},
                                       {"required", json::array()}, {"properties", json::object()}};
    CHECK(cb::check_closed_command_schema(schema, "result.steps", {"result.steps"}).ok());
}

TEST_CASE("hostile values: deep nesting and huge arrays are refused without recursion") {
    const json schema = {{"type", "object"}, {"additionalProperties", true}};

    // 200,000 levels, built by the parser, which does not recurse.
    std::string text = "{\"a\":";
    text += std::string(200000, '[');
    text += std::string(200000, ']');
    text += "}";
    const json deep = json::parse(text);
    const auto err = cb::validate_command_value(schema, deep);
    CHECK(err.code == "command.invalid_args");
    CHECK(err.message == "args is nested more than 64 levels deep.");

    // Exactly 64 levels pass.
    std::string ok_text = std::string(63, '[') + std::string(63, ']');
    const json ok_deep = {{"a", json::parse(ok_text)}};
    CHECK(cb::measure_command_value(ok_deep).ok());
    const json too_deep = {{"a", json::parse("[" + ok_text + "]")}};
    CHECK(cb::measure_command_value(too_deep).code == "command.invalid_args");

    // More than 2^20 values.
    json huge = json::object();
    huge["a"] = json::array();
    huge["a"].get_ref<json::array_t&>().resize(std::size_t(1) << 20);
    CHECK(cb::validate_command_value(schema, huge).message == "args holds more than 1,048,576 values.");
    huge["a"].get_ref<json::array_t&>().resize((std::size_t(1) << 20) - 2);
    CHECK(cb::validate_command_value(schema, huge).ok());

    // A schema the validator was never meant to see does not throw.
    const json broken = {{"type", "array"}, {"minItems", "three"}};
    CHECK(cb::validate_command_value(broken, json::array()).code == "command.bad_schema");
}

// A null on an optional argument counts as its absence (design/command-layer.md,
// Arguments): a local model sent source.add with every optional argument
// null, which validation refused, and it had to try again.
namespace {
const json& args_of(const std::string& name) {
    for (const auto& d : cb::command_table())
        if (d.name == name) return d.args;
    static const json none;
    FAIL("no command " << name);
    return none;
}
}  // namespace

TEST_CASE("a null on an optional argument is dropped before validation, and nothing else is") {
    // The call as a local model sent it: the optional sql, from and transform null.
    json call = {{"id", "layer0"}, {"catalog", "live:entropy.l0"}, {"sql", nullptr}, {"from", nullptr},
                 {"transform", nullptr}, {"dry_run", false}};
    const json& add = args_of("source.add");
    CHECK(fault(add, call) == "args.from");   // refused as it came
    CHECK(cb::drop_null_optional_args(add, &call) == 3);
    CHECK(call == json({{"id", "layer0"}, {"catalog", "live:entropy.l0"}, {"dry_run", false}}));
    CHECK(fault(add, call) == "ok");

    // A null on a required argument stays, and is refused as before.
    json no_id = {{"id", nullptr}, {"catalog", "live:entropy.l0"}};
    CHECK(cb::drop_null_optional_args(add, &no_id) == 0);
    CHECK(no_id["id"].is_null());
    CHECK(fault(add, no_id) == "args.id");

    // Nested: an optional field of an item goes, a required one stays.
    json nested = {{"id", "e"}, {"catalog", "live:attn"},
                   {"transform", json::array({{{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}, {"keepdims", nullptr}},
                                              {{"op", nullptr}, {"axis", nullptr}}})}};
    CHECK(cb::drop_null_optional_args(add, &nested) == 2);
    CHECK(nested["transform"][0] == json({{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}}));
    CHECK(nested["transform"][1] == json({{"op", nullptr}}));
    CHECK(fault(add, nested) == "args.transform[1].op");

    // Where the schema admits null, null is a value and stays: in panel.patch it
    // takes the title away. A null for an object that admits none goes.
    json patch = {{"id", "loss"}, {"patch", {{"title", nullptr}, {"bind", nullptr}, {"options", {{"y_log", nullptr}}}}}};
    const json& panel_patch = args_of("panel.patch");
    CHECK(cb::drop_null_optional_args(panel_patch, &patch) == 1);
    CHECK(patch == json({{"id", "loss"}, {"patch", {{"title", nullptr}, {"options", {{"y_log", nullptr}}}}}}));
    CHECK(fault(panel_patch, patch) == "ok");

    // Schemas by hand: an enum that lists null keeps it, the values of
    // additionalProperties are left alone, and a schema with no type admits null.
    const json s = {{"type", "object"},
                    {"properties", {{"mode", {{"enum", json::array({"a", nullptr})}}},
                                    {"any", json::object()},
                                    {"n", {{"type", "integer"}}},
                                    {"map", {{"type", "object"}, {"additionalProperties", {{"type", "string"}}}}}}}};
    json v = {{"mode", nullptr}, {"any", nullptr}, {"n", nullptr}, {"map", {{"k", nullptr}}}};
    CHECK(cb::drop_null_optional_args(s, &v) == 1);
    CHECK(v == json({{"mode", nullptr}, {"any", nullptr}, {"map", {{"k", nullptr}}}}));
    // Not an object: nothing to do.
    json scalar = nullptr;
    CHECK(cb::drop_null_optional_args(s, &scalar) == 0);
    CHECK(cb::drop_null_optional_args(s, nullptr) == 0);
}
