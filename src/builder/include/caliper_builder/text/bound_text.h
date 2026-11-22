#pragma once
// caliper_builder::text: the format language of bound texts
// (design/instrument-interface.md, section Bound Text).
//
//   text   = { literal | "{{" | "}}" | field }
//   field  = "{" name [ ":" format ] "}"
//   format = [ "," ] [ "." digits ] ( "d" | "f" | "e" | "g" | "%" | "s" ) | "si" [ "." digits ]
//
// A format has at most 1,024 characters and 16 fields, and its output at
// most 1,024 characters, ending with an ellipsis when longer. The language
// has no expressions. It needs no ImGui and no platform code, so that the
// browser's spec runtime can build it to WebAssembly.
//
// The functions are defined in src/builder/text/bound_text.cpp. Nothing
// here throws.
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/error.h>

namespace caliper_builder::text {

// One field of a parsed format.
struct FieldFormat {
    std::string name;
    bool grouping = false;       // ","
    int precision = -1;          // "." digits; -1 when absent
    char type = 's';             // 'd', 'f', 'e', 'g', '%', 's', or 'k' for "si"
};

// A parsed format: literal parts and fields, in order. parts[i] is a
// literal when fields_at[i] is -1, else the field fields[fields_at[i]].
struct ParsedText {
    std::vector<std::string> parts;
    std::vector<int> fields_at;
    std::vector<FieldFormat> fields;
};

// Parses `format` against the names of the fields table. spec.bad_format
// for an unknown name, an open or nested brace, a bad format, more than 16
// fields or more than 1,024 characters.
Error parse_bound_text(std::string_view format, const std::vector<std::string>& field_names, ParsedText* out);

// Formats the values of the fields: a number with its format, a text with
// `s`, a dash for a null value, "?" for a format that does not fit the
// value's type, and the shape "[16]" for an array of more than one element.
std::string format_bound_text(const ParsedText& text, const std::map<std::string, nlohmann::json>& values);

}  // namespace caliper_builder::text
