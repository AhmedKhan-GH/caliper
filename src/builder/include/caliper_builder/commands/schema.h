#pragma once
// The built-in validator for the subset of JSON Schema that command
// descriptors use (design/command-layer.md, The schema subset): type (a
// name or a list), properties, required, additionalProperties (a boolean or a
// schema), items, minItems, maxItems, enum, minimum, maximum, minLength and
// maxLength (in code points), pattern (ECMAScript, searched, on strings of at
// most 4096 bytes), and the annotations description, title and default.
//
// check_command_schema() refuses a schema that uses anything else, nests
// deeper than 32 levels or holds a pattern that does not compile
// ("command.bad_schema"). measure_command_value() walks a value without
// recursion and refuses one nested deeper than 64 levels or holding more than
// 2^20 values; validate_command_value() measures first, then validates, and
// returns the first fault as "command.invalid_args" with `where` set to the
// value's path ("args.bind.y[1]"). Properties are checked in name order, so
// the first fault is deterministic. check_closed_command_schema() adds the
// rules of the descriptor table: every object schema closed, its required
// properties listed and every property described.
//
// Values come from agents and are treated as hostile; schemas come from the
// descriptor table. No function here throws.
#include <caliper_builder/error.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace caliper_builder {

struct CommandSchemaLimits {
    static constexpr std::size_t max_value_depth = 64;
    static constexpr std::size_t max_value_nodes = std::size_t(1) << 20;
    static constexpr std::size_t max_schema_depth = 32;
    static constexpr std::size_t max_pattern_input = 4096;
};

Error check_command_schema(const nlohmann::json& schema);

Error measure_command_value(const nlohmann::json& value, std::string_view root = "args");

Error validate_command_value(const nlohmann::json& schema, const nlohmann::json& value,
                             std::string_view root = "args");

// A null on an optional argument counts as the argument's absence: removes
// from `value`, an object that `schema` describes, every property whose value
// is null when its object schema does not list it in `required` and the
// property's own schema does not admit null (its type is not "null" or a list
// holding it, or its enum does not list null). It follows the schema's
// properties into nested objects and the items of arrays, at most 32 levels,
// and leaves alone a required property, a null the schema admits (such as the
// null of a merge patch, which takes a value away) and the values of
// additionalProperties. Returns how many properties it removed. The registry
// calls it before it validates an invocation, so that the journal, the
// approval cards and the handler see the call as it was understood; clients
// that check arguments first (caliper-mcp) call it too.
std::size_t drop_null_optional_args(const nlohmann::json& schema, nlohmann::json* value);

// `open_paths` names the object schemas allowed to stay open, as paths such as
// "result.json" (properties joined by dots, "[]" for items).
Error check_closed_command_schema(const nlohmann::json& schema, std::string_view root,
                                  const std::vector<std::string>& open_paths = {});

// The length of a UTF-8 string in code points, as minLength and maxLength
// count it.
std::size_t command_code_points(std::string_view text);

}  // namespace caliper_builder
