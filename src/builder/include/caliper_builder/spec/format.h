#pragma once
// Reading and writing a spec in its two forms: TOML, the source of truth on
// disk, and the canonical JSON form, which the JSON Schema describes and the
// browser reads (design/spec-format.md).
//
// Parsing checks the text, the values (no dates, no null, finite numbers, safe
// integers, valid UTF-8), the size and nesting of the document, the types, the
// keys of each table, the required keys and the enumerations. It does not check
// references, shapes or value ranges; validate() in validate.h does. A parse
// that fails returns its errors and an empty Spec. Errors of `parse_toml` carry
// the line and column of the value they concern.
//
// The writers are deterministic: parsing what they write gives back an equal
// Spec, and writing that Spec gives the same text. The TOML writer keeps no
// comments. None of these functions throws.
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include <caliper_builder/spec/spec.h>

namespace caliper_builder::spec {

struct Position {
    int line = 0;
    int column = 0;
};

// Document path ("source[1].transform[0].axis") to the position of its value
// in the TOML text.
using SourceMap = std::map<std::string, Position>;

struct ParseResult {
    Spec spec;
    std::vector<Diagnostic> diagnostics;  // errors only
    SourceMap positions;                  // filled by parse_toml only
    bool ok() const { return !has_errors(diagnostics); }
};

ParseResult parse_toml(std::string_view text);
ParseResult from_json(const nlohmann::json& document);
ParseResult from_json_text(std::string_view text);

std::string to_toml(const Spec& spec);
nlohmann::json to_json(const Spec& spec);
// The canonical JSON text: compact UTF-8 with keys in byte order.
std::string to_json_text(const Spec& spec);

// One item of a spec, decoded from its JSON form with the same checks as a
// whole document. `path` prefixes the paths of the diagnostics, for example
// "panel[3]". The command layer uses these for panel.create, source.add and
// layout.set.
template <class T>
struct Decoded {
    T value{};
    std::vector<Diagnostic> diagnostics;
    bool ok() const { return !has_errors(diagnostics); }
};

Decoded<Source> source_from_json(const nlohmann::json& item, const std::string& path = "source");
Decoded<Panel> panel_from_json(const nlohmann::json& item, const std::string& path = "panel");
Decoded<LayoutNode> layout_from_json(const nlohmann::json& item, const std::string& path = "layout");
// One control (control.h), for control.create and control.patch.
Decoded<Control> control_from_json(const nlohmann::json& item, const std::string& path = "control");
// One program (program.h), for program.create and program.patch, with the
// checks of a whole document.
Decoded<Program> program_from_json(const nlohmann::json& item, const std::string& path = "program");
// What the reader gives for each item of the top-level array `stage` of an old spec, because
// pieces replaced stages: the warning spec.stage_replaced at the stage's path. The spec still
// opens, without the stage. Its message names the
// stage's kind, entity and panel (the keys of `stage`, any of them absent) and says how to write the
// same look as a piece program.
Diagnostic stage_replaced(const nlohmann::json& stage, const std::string& path);
// One state variable, one entity of a scene panel and one annotation of a
// panel (spec version 1, state.h, entity.h and annotation.h), for the state,
// entity and annotation commands. An entity or an annotation reads as an item
// of a panel, so that the path "panel[2].entity[0]" prefixes its diagnostics.
Decoded<StateVar> state_from_json(const nlohmann::json& item, const std::string& path = "state");
Decoded<Entity> entity_from_json(const nlohmann::json& item, const std::string& path = "entity");
Decoded<Annotation> annotation_from_json(const nlohmann::json& item, const std::string& path = "annotation");

nlohmann::json to_json(const Source& source);
nlohmann::json to_json(const Panel& panel);
nlohmann::json to_json(const LayoutNode& node);
nlohmann::json to_json(const Control& control);
// The canonical JSON of a program: geometry outputs always carry part and
// pixels outputs always carry format, and only the keys of the output's kind
// are written. The provenance table of a spec that is not empty.
nlohmann::json to_json(const Program& program);
nlohmann::json to_json(const Provenance& provenance);
// The canonical JSON of a state variable, an entity and an annotation.
nlohmann::json to_json(const StateVar& state);
nlohmann::json to_json(const Entity& entity);
nlohmann::json to_json(const Annotation& annotation);

// Fills the line and column of each diagnostic that has none from the
// positions of a parsed TOML text: the position of its path, or of the nearest
// enclosing value that has one.
void locate(std::vector<Diagnostic>& diagnostics, const SourceMap& positions);

}  // namespace caliper_builder::spec
