// The spec tests of annotations, bound texts and layout shares
// (design/instrument-interface.md, Bound Text, Annotations and Markers, Layout Shares).
#include <doctest/doctest.h>

#include <caliper_builder/spec/format.h>
#include <caliper_builder/spec/validate.h>

#include "helpers.h"

#include <string>

using namespace spec_test;

namespace {

// A version 1 spec with a table source, a state, a range control and a
// series panel `loss`; `panel_tail` follows the panel's keys, and `tail`
// follows everything.
std::string doc(const std::string& panel_tail, const std::string& tail = "") {
    return "caliper_spec = 1\n"                               // 1
           "[applet]\n"                                       // 2
           "id = \"a\"\n"                                     // 3
           "[[source]]\n"                                     // 4
           "id = \"m\"\n"                                     // 5
           "catalog = \"table:run42_metrics\"\n"              // 6
           "[[state]]\n"                                      // 7
           "id = \"token\"\n"                                 // 8
           "type = \"integer\"\n"                             // 9
           "[[state]]\n"                                      // 10
           "id = \"name\"\n"                                  // 11
           "type = \"text\"\n"                                // 12
           "[[control]]\n"                                    // 13
           "id = \"window\"\n"                                // 14
           "kind = \"range\"\n"                               // 15
           "bind = \"value\"\n"                               // 16
           "options = { min = 0.0, max = 100.0, value = [10, 20] }\n"   // 17
           "[[panel]]\n"                                      // 18
           "id = \"loss\"\n"                                  // 19
           "primitive = \"series\"\n"                         // 20
           "bind = { x = \"m.step\", y = \"m.loss\" }\n" +    // 21
           panel_tail + tail;                                 // 22 on
}

// The diagnostics of `text` that these checks give: every error and warning,
// except spec.not_built and spec.unsupported_control at a path outside
// annotations, legends, scale bars, bound texts and layout.
std::vector<cs::Diagnostic> ours(const std::string& text) {
    const cs::ValidationReport r = cs::check_toml(text, fixture_resolver());
    std::vector<cs::Diagnostic> out;
    for (const cs::Diagnostic& d : r.diagnostics) {
        const bool mine = d.path.find(".annotation") != std::string::npos || d.path.find(".legend") != std::string::npos ||
                          d.path.find(".scale_bar") != std::string::npos || d.path.find(".format") != std::string::npos ||
                          d.path.find(".fields") != std::string::npos || d.path.rfind("layout", 0) == 0;
        if ((d.code == "spec.not_built" || d.code == "spec.unsupported_control") && !mine) continue;
        out.push_back(d);
    }
    return out;
}

int line_of(const std::vector<cs::Diagnostic>& ds, const std::string& code) {
    const cs::Diagnostic* d = find(ds, code);
    return d ? d->line : -1;
}

}  // namespace

TEST_CASE("the four annotation kinds of a series panel are valid") {
    const auto ds = ours(doc("[[panel.annotation]]\n"
                             "id = \"window\"\n"
                             "kind = \"band\"\n"
                             "axis = \"x\"\n"
                             "range = { control = \"window\" }\n"
                             "text = \"Window\"\n"
                             "[[panel.annotation]]\n"
                             "id = \"floor\"\n"
                             "kind = \"threshold\"\n"
                             "axis = \"y\"\n"
                             "value = 0.25\n"
                             "emphasis = \"muted\"\n"
                             "[[panel.annotation]]\n"
                             "id = \"peak\"\n"
                             "kind = \"label\"\n"
                             "at = [40.0, { state = \"token\", default = 0 }]\n"
                             "text = \"Loss {v:.3f} at token {t:d}\"\n"
                             "fields = { v = \"m.loss\", t = { state = \"token\" } }\n"
                             "leader = false\n"
                             "[[panel.annotation]]\n"
                             "id = \"now\"\n"
                             "kind = \"marker\"\n"
                             "at = [{ state = \"cursor\", field = \"step\" }, 0.5]\n"
                             "[[panel.annotation]]\n"
                             "id = \"lo-hi\"\n"
                             "kind = \"band\"\n"
                             "axis = \"y\"\n"
                             "range = [{ control = \"window\", field = \"lo\" }, 1.0]\n"));
    CHECK_MESSAGE(ds.empty(), dump(ds));
}

TEST_CASE("an annotation's keys follow its kind") {
    const auto ds = ours(doc("[[panel.annotation]]\n"      // 22
                             "id = \"t\"\n"                // 23
                             "kind = \"threshold\"\n"      // 24
                             "axis = \"y\"\n"              // 25
                             "value = 1.0\n"               // 26
                             "at = [1.0, 2.0]\n"           // 27
                             "[[panel.annotation]]\n"      // 28
                             "id = \"b\"\n"                // 29
                             "kind = \"band\"\n"           // 30
                             "axis = \"x\"\n"));           // 31
    CHECK(has(ds, "spec.unknown_option", "panel[0].annotation[0].at"));
    CHECK(line_of(ds, "spec.unknown_option") == 27);
    CHECK(has(ds, "spec.missing_key", "panel[0].annotation[1].range"));
    CHECK(has(ds, "spec.unknown_option") + has(ds, "spec.missing_key") == 2);
    const auto label = ours(doc("[[panel.annotation]]\nid = \"l\"\nkind = \"label\"\nat = [1.0, 2.0]\n"
                                "axis = \"x\"\nleader = true\n"));
    CHECK(has(label, "spec.missing_key", "panel[0].annotation[0].text"));
    CHECK(has(label, "spec.unknown_option", "panel[0].annotation[0].axis"));
    const auto marker = ours(doc("[[panel.annotation]]\nid = \"m\"\nkind = \"marker\"\nat = [1.0, 2.0]\nleader = true\n"));
    CHECK(has(marker, "spec.unknown_option", "panel[0].annotation[0].leader"));
}

TEST_CASE("ids, axes, emphasis, positions and the count of annotations") {
    const auto ds = ours(doc("[[panel.annotation]]\nid = \"a\"\nkind = \"threshold\"\naxis = \"z\"\nvalue = 1.0\n"
                             "emphasis = \"loud\"\n"
                             "[[panel.annotation]]\nid = \"a\"\nkind = \"marker\"\nat = [1.0, 2.0, 3.0]\n"
                             "[[panel.annotation]]\nid = \"B\"\nkind = \"band\"\naxis = \"x\"\nrange = [5.0, 1.0]\n"));
    CHECK(has(ds, "spec.bad_value", "panel[0].annotation[0].axis"));
    CHECK(has(ds, "spec.bad_value", "panel[0].annotation[0].emphasis"));
    CHECK(has(ds, "spec.duplicate_id", "panel[0].annotation[1].id"));
    CHECK(has(ds, "spec.wrong_type", "panel[0].annotation[1].at"));
    CHECK(has(ds, "spec.bad_id", "panel[0].annotation[2].id"));
    CHECK(has(ds, "spec.bad_value", "panel[0].annotation[2].range"));

    std::string many;
    for (int i = 0; i < 65; ++i)
        many += "[[panel.annotation]]\nid = \"t" + std::to_string(i) + "\"\nkind = \"threshold\"\naxis = \"y\"\nvalue = 1.0\n";
    CHECK(has(ours(doc(many)), "spec.too_many", "panel[0].annotation"));
}

TEST_CASE("the references of annotations and their refusals") {
    const auto ds = ours(doc("[[panel.annotation]]\nid = \"a\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { control = \"window\" }\n"                                   // a pair
                             "[[panel.annotation]]\nid = \"b\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { control = \"nope\" }\n"
                             "[[panel.annotation]]\nid = \"c\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { state = \"nope\" }\n"
                             "[[panel.annotation]]\nid = \"d\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { state = \"cursor\", field = \"bogus\" }\n"
                             "[[panel.annotation]]\nid = \"e\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { state = \"token\", control = \"window\" }\n"
                             "[[panel.annotation]]\nid = \"f\"\nkind = \"band\"\naxis = \"x\"\n"
                             "range = { state = \"token\" }\n"
                             "[[panel.annotation]]\nid = \"g\"\nkind = \"marker\"\n"
                             "at = { selection = \"loss\" }\n"
                             "[[panel.annotation]]\nid = \"h\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { state = \"token\", default = \"x\" }\n"
                             "[[panel.annotation]]\nid = \"i\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { state = \"token\", extra = 1 }\n"
                             "[[panel.annotation]]\nid = \"j\"\nkind = \"threshold\"\naxis = \"y\"\n"
                             "value = { control = \"window\", field = \"mid\" }\n"));
    CHECK(has(ds, "spec.reference_type", "panel[0].annotation[0].value"));
    CHECK(has(ds, "spec.unknown_control", "panel[0].annotation[1].value"));
    CHECK(has(ds, "spec.unknown_state", "panel[0].annotation[2].value"));
    CHECK(has(ds, "spec.unknown_field", "panel[0].annotation[3].value"));
    CHECK(has(ds, "spec.bad_reference", "panel[0].annotation[4].value"));
    CHECK(has(ds, "spec.reference_type", "panel[0].annotation[5].range"));
    CHECK(has(ds, "spec.reference_type", "panel[0].annotation[6].at"));
    CHECK(has(ds, "spec.bad_reference", "panel[0].annotation[7].value.default"));
    CHECK(has(ds, "spec.bad_reference", "panel[0].annotation[8].value"));
    CHECK(has(ds, "spec.unknown_field", "panel[0].annotation[9].value"));
}

TEST_CASE("a bound text in an annotation, a text panel and a readout") {
    // The format's own refusals, at the line of the format.
    const auto open = ours(doc("[[panel.annotation]]\n"        // 22
                               "id = \"l\"\n"                  // 23
                               "kind = \"label\"\n"            // 24
                               "at = [1.0, 2.0]\n"             // 25
                               "text = \"Loss {v:.3f\"\n"      // 26
                               "fields = { v = \"m.loss\" }\n"));
    CHECK(has(open, "spec.bad_format", "panel[0].annotation[0].text"));
    CHECK(line_of(open, "spec.bad_format") == 26);

    const auto unknown = ours(doc("[[panel.annotation]]\nid = \"l\"\nkind = \"label\"\nat = [1.0, 2.0]\n"
                                  "text = \"{w}\"\nfields = { v = \"m.loss\" }\n"));
    CHECK(has(unknown, "spec.bad_format", "panel[0].annotation[0].text"));
    // A number format over a text state is refused; s fits it.
    const auto text_state = ours(doc("[[panel.annotation]]\nid = \"l\"\nkind = \"label\"\nat = [1.0, 2.0]\n"
                                     "text = \"{n:.2f}\"\nfields = { n = { state = \"name\" } }\n"));
    CHECK(has(text_state, "spec.bad_format", "panel[0].annotation[0].text"));
    CHECK(ours(doc("[[panel.annotation]]\nid = \"l\"\nkind = \"label\"\nat = [1.0, 2.0]\n"
                   "text = \"{n:s}\"\nfields = { n = { state = \"name\" } }\n"))
              .empty());
    const auto no_source = ours(doc("[[panel.annotation]]\nid = \"l\"\nkind = \"label\"\nat = [1.0, 2.0]\n"
                                    "text = \"{v}\"\nfields = { v = \"gone\", \"bad name\" = \"m\" }\n"));
    CHECK(has(no_source, "spec.unknown_source", "panel[0].annotation[0].fields.v"));
    CHECK(has(no_source, "spec.bad_value"));

    // A text panel: a bound value is the field `value`, and text excludes format.
    const std::string text_panel = "[[panel]]\nid = \"now\"\nprimitive = \"text\"\nbind = { value = \"m.loss\" }\n";
    // The panel draws a correct one at each frame, so it has no diagnostic.
    const auto bound = ours(doc("", text_panel + "options = { format = \"Loss {value:.3f} at {s:,d}\", fields = { s = "
                                                 "{ state = \"cursor\", field = \"step\" } } }\n"));
    CHECK_MESSAGE(bound.empty(), dump(bound));
    CHECK(has(ours(doc("", text_panel + "options = { text = \"Loss\", format = \"{value}\" }\n")), "spec.bad_value",
              "panel[1].options.format"));
    CHECK(has(ours(doc("", text_panel + "options = { format = \"{value}\", fields = { value = \"m.loss\" } }\n")),
              "spec.bad_value", "panel[1].options.fields.value"));

    // A readout needs its format.
    const std::string readout = "[[control]]\nid = \"now\"\nkind = \"readout\"\n";
    CHECK(has(ours(doc("", readout)), "spec.missing_key", "control[1].options.format"));
    CHECK_FALSE(has(ours(doc("", readout + "options = { format = \"Step {s:,d}\", fields = { s = { state = \"cursor\", "
                                           "field = \"step\" } } }\n")),
                    "spec.missing_key"));
    std::string seventeen = "options = { format = \"x\", fields = { ";
    for (int i = 0; i < 17; ++i) seventeen += (i ? ", " : "") + std::string("f") + std::to_string(i) + " = \"m\"";
    CHECK(has(ours(doc("", readout + seventeen + " } }\n")), "spec.bad_format", "control[1].options.fields"));
}

TEST_CASE("legends and scale bars") {
    CHECK(ours(doc("options = { legend = \"auto\" }\n")).empty());
    CHECK(ours(doc("options = { legend = false }\n")).empty());
    CHECK(ours(doc("options = { legend = true }\n")).empty());
    CHECK(has(ours(doc("options = { legend = \"maybe\" }\n")), "spec.bad_value", "panel[0].options.legend"));
    const std::string heat = "[[panel]]\nid = \"h\"\nprimitive = \"heatmap\"\nbind = { values = \"w\" }\n";
    const std::string tensor = "[[source]]\nid = \"w\"\ncatalog = \"tensor:attn_weights\"\n";
    const auto ds = ours(doc("", tensor + heat + "options = { legend = true }\n"));
    CHECK(has(ds, "spec.not_built", "panel[1].options.legend"));
    const std::string image = "[[panel]]\nid = \"i\"\nprimitive = \"image\"\nbind = { values = \"w\" }\n";
    CHECK(has(ours(doc("", tensor + image + "options = { scale_bar = true }\n")), "spec.not_built",
              "panel[1].options.scale_bar"));
}

TEST_CASE("annotations outside a series panel") {
    const std::string tensor = "[[source]]\nid = \"w\"\ncatalog = \"tensor:attn_weights\"\n";
    const std::string heat = "[[panel]]\nid = \"h\"\nprimitive = \"heatmap\"\nbind = { values = \"w\" }\n"
                             "[[panel.annotation]]\nid = \"t\"\nkind = \"threshold\"\naxis = \"y\"\nvalue = 1.0\n";
    CHECK(has(ours(doc("", tensor + heat)), "spec.not_built", "panel[1].annotation"));
    const std::string text = "[[panel]]\nid = \"t\"\nprimitive = \"text\"\noptions = { text = \"x\" }\n"
                             "[[panel.annotation]]\nid = \"t\"\nkind = \"threshold\"\naxis = \"y\"\nvalue = 1.0\n";
    CHECK(has(ours(doc("", text)), "spec.unknown_option", "panel[1].annotation"));
}

TEST_CASE("layout shares and leaf minimums") {
    const std::string p2 = "[[panel]]\nid = \"two\"\nprimitive = \"series\"\nbind = { x = \"m.step\", y = \"m.loss\" }\n";
    const std::string p3 = "[[panel]]\nid = \"three\"\nprimitive = \"series\"\nbind = { x = \"m.step\", y = \"m.loss\" }\n";
    const auto fine = ours(doc("", p2 + p3 +
                                       "[layout]\nsplit = \"vertical\"\nshares = [6, 3, 1, 1]\nchildren = [{ panel = \"loss\", "
                                       "min = [48, 20] }, \"two\", \"three\", \"window\"]\n"));
    CHECK_MESSAGE(fine.empty(), dump(fine));
    const auto both = ours(doc("", p2 + "[layout]\nsplit = \"vertical\"\nratio = 0.3\nshares = [1, 2]\nchildren = "
                                        "[\"loss\", \"two\"]\n"));
    CHECK(has(both, "spec.bad_layout", "layout.shares"));
    const auto count = ours(doc("", p2 + "[layout]\nsplit = \"vertical\"\nshares = [1, 2, 3]\nchildren = [\"loss\", "
                                         "\"two\"]\n"));
    CHECK(has(count, "spec.bad_layout", "layout.shares"));
    const auto zero = ours(doc("", p2 + "[layout]\nsplit = \"vertical\"\nshares = [0, 2]\nchildren = [\"loss\", \"two\"]\n"));
    CHECK(has(zero, "spec.out_of_range", "layout.shares[0]"));
    const auto small = ours(doc("", p2 + "[layout]\nsplit = \"vertical\"\nchildren = [{ panel = \"loss\", min = [3, "
                                         "401] }, \"two\"]\n"));
    CHECK(has(small, "spec.out_of_range", "layout.children[0].min[0]"));
    CHECK(has(small, "spec.out_of_range", "layout.children[0].min[1]"));
}
