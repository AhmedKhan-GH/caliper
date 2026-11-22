// The spec tests of state variables, binding references and the new control kinds
// (design/instrument-interface.md, Binding References, Shared State and Controls).
#include <doctest/doctest.h>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/validate.h>

#include <string>

#include "helpers.h"

using namespace spec_test;

namespace {

const char* kHead = "caliper_spec = 1\n[applet]\nid = \"a\"\n";

cs::ValidationReport check(const std::string& body) { return cs::check_toml(kHead + body); }

// A state table of the given type; `extra` holds more keys, one per line.
std::string state(const std::string& id, const std::string& type, const std::string& extra = "") {
    return "[[state]]\nid = \"" + id + "\"\ntype = \"" + type + "\"\n" + (extra.empty() ? "" : extra + "\n");
}

std::string control(const std::string& id, const std::string& kind, const std::string& bind,
                    const std::string& options = "") {
    std::string s = "[[control]]\nid = \"" + id + "\"\nkind = \"" + kind + "\"\nbind = \"" + bind + "\"\n";
    if (!options.empty()) s += "options = { " + options + " }\n";
    return s;
}

const char* kToken = "[[state]]\nid = \"token\"\ntype = \"integer\"\ndefault = 0\nmin = 0\nmax = 63\n";
const char* kMeasure = "[[state]]\nid = \"measure\"\ntype = \"text\"\ndefault = \"norm\"\nchoices = [\"norm\", \"mean\"]\n";

// A source `s` whose slice index is the TOML text `index`.
std::string slice_source(const std::string& index) {
    return "[[source]]\nid = \"s\"\ncatalog = \"live:x\"\n  [[source.transform]]\n  op = \"slice\"\n  axis = 0\n  index = " +
           index + "\n";
}

const char* kScenePanel = "[[panel]]\nid = \"world\"\nprimitive = \"scene\"\nbind = { scene = \"live:model\" }\n";

std::string scene_select(const std::string& select) {
    return "[[panel]]\nid = \"world\"\nprimitive = \"scene\"\nbind = { scene = \"live:model\" }\noptions = { select = " +
           select + " }\n";
}

// The 1-based line of the n-th (from 0) line of `text` that starts with `needle`.
int line_of(const std::string& text, const std::string& needle, int n = 0) {
    int line = 1;
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t end = text.find('\n', pos);
        if (end == std::string::npos) end = text.size();
        if (text.compare(pos, needle.size(), needle) == 0 && n-- == 0) return line;
        pos = end + 1;
        ++line;
    }
    return -1;
}

std::string two_choices(const std::string& k1, const std::string& k2) {
    return "choices = [{ value = \"norm\", key = \"" + k1 + "\" }, { value = \"mean\", key = \"" + k2 + "\" }]";
}

}  // namespace

// A full instrument with a state, a ranged follow control and every new control kind passes.
TEST_CASE("references spec: a valid instrument has no error") {
    const std::string body =
        std::string(kToken) + kMeasure + slice_source("{ state = \"token\" }") +
        "[[panel]]\nid = \"h\"\nprimitive = \"heatmap\"\nbind = { values = \"s\" }\n" +
        control("window", "range", "value",
                "min = { state = \"cursor\", field = \"first\" }, max = { state = \"cursor\", field = \"latest\" }, "
                "value = [0, 2048], min_span = 16, follow = true") +
        control("measure", "segmented", "state:measure", two_choices("n", "m")) +
        control("pick", "picker", "value", "choices = [\"a\", \"b\"], value = \"a\"") +
        control("name", "text", "value", "placeholder = \"Name\", max_length = 32") +
        control("flag", "toggle", "value", "value = true");
    auto r = check(body);
    CHECK_MESSAGE(r.ok(), dump(r.diagnostics));
}

// Each state rule reports its code at the path of the item, and the path maps to the TOML line.
TEST_CASE("references spec: state declarations") {
    // A duplicate id, at the second table's id line.
    const std::string dup = state("t", "number") + state("t", "number");
    auto d = check(dup);
    CHECK_MESSAGE(has(d.diagnostics, "spec.duplicate_id", "state[1].id"), dump(d.diagnostics));
    REQUIRE(find(d.diagnostics, "spec.duplicate_id"));
    CHECK(find(d.diagnostics, "spec.duplicate_id")->line == line_of(std::string(kHead) + dup, "id = \"t\"", 1));

    auto reserved = check(state("cursor", "number"));
    CHECK_MESSAGE(has(reserved.diagnostics, "spec.bad_id", "state[0].id"), dump(reserved.diagnostics));

    // The default is out of the bounds.
    const std::string big = state("t", "integer", "default = 99\nmin = 0\nmax = 63");
    auto bounds = check(big);
    CHECK_MESSAGE(has(bounds.diagnostics, "spec.out_of_range", "state[0].default"), dump(bounds.diagnostics));
    REQUIRE(find(bounds.diagnostics, "spec.out_of_range"));
    CHECK(find(bounds.diagnostics, "spec.out_of_range")->line == line_of(std::string(kHead) + big, "default = 99"));

    auto choice = check(state("t", "text", "default = \"x\"\nchoices = [\"a\", \"b\"]"));
    CHECK_MESSAGE(has(choice.diagnostics, "spec.out_of_range", "state[0].default"), dump(choice.diagnostics));

    auto order = check(state("t", "number", "min = 5\nmax = 5"));
    CHECK_MESSAGE(has(order.diagnostics, "spec.out_of_range", "state[0].min"), dump(order.diagnostics));

    auto boolean = check(state("t", "boolean", "choices = [\"a\"]"));
    CHECK_MESSAGE(has(boolean.diagnostics, "spec.bad_value", "state[0].choices"), dump(boolean.diagnostics));

    std::string many;
    for (int i = 0; i < 65; ++i) many += state("s" + std::to_string(i), "number");
    auto too_many = check(many);
    CHECK_MESSAGE(has(too_many.diagnostics, "spec.too_many", "state"), dump(too_many.diagnostics));
}

// Controls that write a state variable must match its type, and each control kind checks its own options.
TEST_CASE("references spec: control codes") {
    const std::string states = std::string(kToken) + kMeasure;
    const std::string two = two_choices("n", "m");

    // A slider cannot write a text state.
    const std::string slider = states + control("k", "slider", "state:measure", "min = 0, max = 1");
    auto mismatch = check(slider);
    CHECK_MESSAGE(has(mismatch.diagnostics, "spec.control_mismatch", "control[0].bind"), dump(mismatch.diagnostics));
    REQUIRE(find(mismatch.diagnostics, "spec.control_mismatch"));
    CHECK(find(mismatch.diagnostics, "spec.control_mismatch")->line ==
          line_of(std::string(kHead) + slider, "bind = \"state:measure\""));

    auto unknown = check(states + control("k", "slider", "state:nope", "min = 0, max = 1"));
    CHECK_MESSAGE(has(unknown.diagnostics, "spec.unknown_state", "control[0].bind"), dump(unknown.diagnostics));

    // Two controls on one state give a warning, and the report stays ok.
    auto writers = check(states + control("a", "slider", "state:token", "min = 0, max = 63") +
                         control("b", "slider", "state:token", "min = 0, max = 63"));
    CHECK_MESSAGE(has(writers.diagnostics, "spec.state_writers"), dump(writers.diagnostics));
    CHECK_MESSAGE(writers.ok(), dump(writers.diagnostics));

    auto value = check(states + control("k", "slider", "state:token", "min = 0, max = 63, value = 3"));
    CHECK_MESSAGE(has(value.diagnostics, "spec.control_mismatch", "control[0].options.value"), dump(value.diagnostics));

    // A segmented group has 2 to 16 choices.
    auto one = check(states + control("k", "segmented", "state:measure", "choices = [\"norm\"]"));
    CHECK_MESSAGE(has(one.diagnostics, "spec.out_of_range", "control[0].options.choices"), dump(one.diagnostics));
    std::string seventeen = "choices = [";
    for (int i = 0; i < 17; ++i) seventeen += std::string(i ? ", " : "") + "\"c" + std::to_string(i) + "\"";
    auto many = check(control("k", "segmented", "value", seventeen + "]"));
    CHECK_MESSAGE(has(many.diagnostics, "spec.out_of_range", "control[0].options.choices"), dump(many.diagnostics));

    // Shortcut keys.
    auto dup = check(control("g1", "segmented", "value", two) + control("g2", "segmented", "value", two));
    CHECK_MESSAGE(has(dup.diagnostics, "spec.duplicate_shortcut"), dump(dup.diagnostics));
    for (const char* key : {"f", "shift+u"}) {
        CAPTURE(key);
        auto r = check(control("g", "segmented", "value", two_choices(key, "m")));
        CHECK_MESSAGE(has(r.diagnostics, "spec.reserved_shortcut", "control[0].options.choices[0].key"), dump(r.diagnostics));
    }
    for (const char* key : {"ctrl+a", "ab"}) {
        CAPTURE(key);
        auto r = check(control("g", "segmented", "value", two_choices(key, "m")));
        CHECK_MESSAGE(has(r.diagnostics, "spec.bad_value", "control[0].options.choices[0].key"), dump(r.diagnostics));
    }
    auto picker_key = check(control("p", "picker", "value", "choices = [{ value = \"a\", key = \"a\" }, \"b\"]"));
    CHECK_MESSAGE(has(picker_key.diagnostics, "spec.control_mismatch"), dump(picker_key.diagnostics));

    auto no_choices = check(control("p", "picker", "value"));
    CHECK_MESSAGE(has(no_choices.diagnostics, "spec.missing_key", "control[0].options.choices"), dump(no_choices.diagnostics));

    // Range options.
    auto no_max = check(control("r", "range", "value", "min = 0"));
    CHECK_MESSAGE(has(no_max.diagnostics, "spec.missing_key", "control[0].options.max"), dump(no_max.diagnostics));
    auto span = check(control("s", "slider", "value", "min = 0, max = 9, min_span = 2"));
    CHECK_MESSAGE(has(span.diagnostics, "spec.control_mismatch", "control[0].options.min_span"), dump(span.diagnostics));
    auto backwards = check(control("r", "range", "value", "min = 0, max = 9, value = [5, 1]"));
    CHECK_MESSAGE(has(backwards.diagnostics, "spec.bad_value", "control[0].options.value"), dump(backwards.diagnostics));

    auto multiple = check(states + control("k", "segmented", "state:measure", two + ", multiple = true"));
    CHECK_MESSAGE(has(multiple.diagnostics, "spec.control_mismatch", "control[0].options.multiple"), dump(multiple.diagnostics));
}

// The option select of a scene panel names a state variable and a pick field that fits it.
TEST_CASE("references spec: select option") {
    auto good = check(std::string(kToken) + scene_select("{ state = \"token\", field = \"column\" }"));
    CHECK_MESSAGE(good.ok(), dump(good.diagnostics));

    auto field = check(std::string(kToken) + scene_select("{ state = \"token\", field = \"colour\" }"));
    CHECK_MESSAGE(has(field.diagnostics, "spec.unknown_field", "panel[0].options.select.field"), dump(field.diagnostics));

    auto state_missing = check(scene_select("{ state = \"nope\", field = \"column\" }"));
    CHECK_MESSAGE(has(state_missing.diagnostics, "spec.unknown_state", "panel[0].options.select.state"),
                  dump(state_missing.diagnostics));

    auto type = check(std::string(kToken) + scene_select("{ state = \"token\", field = \"x\" }"));
    CHECK_MESSAGE(has(type.diagnostics, "spec.reference_type", "panel[0].options.select.field"), dump(type.diagnostics));

    auto heatmap = check(std::string(kToken) + slice_source("0") +
                         "[[panel]]\nid = \"h\"\nprimitive = \"heatmap\"\nbind = { values = \"s\" }\n"
                         "options = { select = { state = \"token\", field = \"column\" } }\n");
    CHECK_MESSAGE(has(heatmap.diagnostics, "spec.unknown_option"), dump(heatmap.diagnostics));
}

// Each reference rule reports its code at the path of the reference, and the path maps to the TOML line.
TEST_CASE("references spec: reference codes") {
    const std::string states = std::string(kToken) + kMeasure + kScenePanel;
    const std::string at = "source[0].transform[0].index";
    auto run = [&](const std::string& index) { return check(states + slice_source(index)); };

    auto no_state = run("{ state = \"nope\" }");
    CHECK_MESSAGE(has(no_state.diagnostics, "spec.unknown_state", at + ".state"), dump(no_state.diagnostics));
    auto no_control = run("{ control = \"nope\" }");
    CHECK_MESSAGE(has(no_control.diagnostics, "spec.unknown_control", at + ".control"), dump(no_control.diagnostics));
    auto lo = run("{ state = \"token\", field = \"lo\" }");
    CHECK_MESSAGE(has(lo.diagnostics, "spec.unknown_field", at + ".field"), dump(lo.diagnostics));
    auto cursor = run("{ state = \"cursor\", field = \"nope\" }");
    CHECK_MESSAGE(has(cursor.diagnostics, "spec.unknown_field", at + ".field"), dump(cursor.diagnostics));

    // A text state at an integer place; the path maps to the line of the index key.
    auto text = run("{ state = \"measure\" }");
    CHECK_MESSAGE(has(text.diagnostics, "spec.reference_type", at), dump(text.diagnostics));
    REQUIRE(find(text.diagnostics, "spec.reference_type"));
    CHECK(find(text.diagnostics, "spec.reference_type")->line ==
          line_of(std::string(kHead) + states + slice_source("{ state = \"measure\" }"), "  index = "));

    auto fallback = run("{ state = \"token\", default = \"x\" }");
    CHECK_MESSAGE(has(fallback.diagnostics, "spec.bad_reference", at + ".default"), dump(fallback.diagnostics));
    auto entity = run("{ state = \"token\", entity = \"e\" }");
    CHECK_MESSAGE(has(entity.diagnostics, "spec.bad_reference"), dump(entity.diagnostics));
    auto panel = run("{ selection = \"nope\", field = \"column\" }");
    CHECK_MESSAGE(has(panel.diagnostics, "spec.unknown_panel", at + ".selection"), dump(panel.diagnostics));
}

// A cycle from a control's options back to its own value is an error, with or without a state in between.
TEST_CASE("references spec: reference cycles") {
    auto self = check(control("window", "range", "value",
                              "min = { control = \"window\", field = \"lo\" }, max = 10"));
    CHECK_MESSAGE(has(self.diagnostics, "spec.reference_cycle", "control[0].options"), dump(self.diagnostics));

    auto through_state = check(state("w", "range", "min = 0\nmax = 10") +
                               control("win", "range", "state:w", "max = { state = \"w\", field = \"hi\" }"));
    CHECK_MESSAGE(has(through_state.diagnostics, "spec.reference_cycle"), dump(through_state.diagnostics));
}

// References in the options and the slots of a panel are not applied yet.
TEST_CASE("references spec: panel references are not built") {
    const std::string base = control("k", "slider", "value", "min = 0, max = 9") + slice_source("0");
    auto option = check(base + "[[panel]]\nid = \"h\"\nprimitive = \"heatmap\"\nbind = { values = \"s\" }\n"
                               "options = { range = { control = \"k\" } }\n");
    CHECK_MESSAGE(has(option.diagnostics, "spec.not_built"), dump(option.diagnostics));

    auto slot = check(state("s", "text", "default = \"a\"") + slice_source("0") +
                      "[[panel]]\nid = \"h\"\nprimitive = \"heatmap\"\nbind = { values = { state = \"s\" } }\n");
    CHECK_MESSAGE(has(slot.diagnostics, "spec.not_built"), dump(slot.diagnostics));
}

// A window over a source whose own steps read a reference is refused.
TEST_CASE("references spec: reference in history") {
    const std::string body = std::string(kToken) +
                             "[[source]]\nid = \"act\"\ncatalog = \"live:x\"\n  [[source.transform]]\n  op = \"slice\"\n"
                             "  axis = 0\n  index = { state = \"token\" }\n"
                             "[[source]]\nid = \"w\"\nfrom = \"act\"\n  [[source.transform]]\n  op = \"window\"\n  steps = 8\n";
    auto r = check(body);
    CHECK_MESSAGE(has(r.diagnostics, "spec.reference_in_history", "source[1].transform[0]"), dump(r.diagnostics));
}
