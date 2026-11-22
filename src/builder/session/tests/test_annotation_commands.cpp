// The session tests of annotation.set and annotation.remove
// (design/instrument-interface.md, Commands).
#include <doctest/doctest.h>

#include <caliper_builder/session/instrument_commands.h>
#include <caliper_builder/session/spec_edit.h>
#include <caliper_builder/spec/format.h>

using caliper_builder::apply_spec_edit;
using caliper_builder::SpecEdit;
using nlohmann::json;
namespace spec = caliper_builder::spec;

namespace {

spec::Spec base(int version = 1) {
    const std::string text = "caliper_spec = " + std::to_string(version) +
                             "\n[applet]\nid = \"a\"\n[[source]]\nid = \"w\"\ncatalog = \"tensor:w\"\n"
                             "[[panel]]\nid = \"loss\"\nprimitive = \"series\"\nbind = { y = \"w\" }\n"
                             "[[panel]]\nid = \"note\"\nprimitive = \"text\"\noptions = { text = \"Note\" }\n";
    spec::ParseResult r = spec::parse_toml(text);
    REQUIRE(r.ok());
    return r.spec;
}

json threshold(const std::string& id, double value) {
    return json{{"id", id}, {"kind", "threshold"}, {"axis", "y"}, {"value", value}, {"text", "Floor"}};
}

}  // namespace

TEST_CASE("annotation.set adds an annotation, then replaces it in place by id") {
    const spec::Spec s = base();
    SpecEdit a = apply_spec_edit(s, "annotation.set", {{"panel", "loss"}, {"annotation", threshold("floor", 0.1)}});
    REQUIRE_MESSAGE(!a.error, a.error.message);
    REQUIRE(a.spec.panels[0].annotations.size() == 1);
    CHECK(a.spec.panels[0].annotations[0].value == json(0.1));
    CHECK(a.result == json{{"panel", "loss"}});

    SpecEdit b = apply_spec_edit(a.spec, "annotation.set",
                                 {{"panel", "loss"},
                                  {"annotation", {{"id", "band"}, {"kind", "band"}, {"axis", "x"}, {"range", {1, 5}}}}});
    REQUIRE_MESSAGE(!b.error, b.error.message);
    SpecEdit c = apply_spec_edit(b.spec, "annotation.set", {{"panel", "loss"}, {"annotation", threshold("floor", 0.4)}});
    REQUIRE_MESSAGE(!c.error, c.error.message);
    const auto& list = c.spec.panels[0].annotations;
    REQUIRE(list.size() == 2);
    CHECK(list[0].id == "floor");   // kept its place
    CHECK(list[0].value == json(0.4));
    CHECK(list[1].id == "band");
}

TEST_CASE("annotation.set raises a version 0 spec to version 1") {
    const spec::Spec s = base(0);
    REQUIRE(s.version == 0);
    SpecEdit a = apply_spec_edit(s, "annotation.set", {{"panel", "loss"}, {"annotation", threshold("floor", 0.1)}});
    REQUIRE_MESSAGE(!a.error, a.error.message);
    CHECK(a.spec.version == spec::kInstrumentVersion);
}

TEST_CASE("annotation.set refuses an unknown panel, a malformed table and an invalid annotation") {
    const spec::Spec s = base();
    CHECK(apply_spec_edit(s, "annotation.set", {{"panel", "nope"}, {"annotation", threshold("f", 1)}}).error.code ==
          "panel.not_found");
    CHECK(apply_spec_edit(s, "annotation.set", {{"panel", "loss"}, {"annotation", "floor"}}).error.code ==
          "spec.wrong_type");
    CHECK(apply_spec_edit(s, "annotation.set",
                          {{"panel", "loss"}, {"annotation", {{"id", "f"}, {"kind", "arrow"}, {"axis", "y"}}}})
              .error.code == "spec.bad_value");
    // Valid as a table, refused by the validator: a band needs its range.
    CHECK(apply_spec_edit(s, "annotation.set",
                          {{"panel", "loss"}, {"annotation", {{"id", "b"}, {"kind", "band"}, {"axis", "x"}}}})
              .error.code == "spec.missing_key");
    // A text panel takes no annotations.
    CHECK(apply_spec_edit(s, "annotation.set", {{"panel", "note"}, {"annotation", threshold("f", 1)}}).error.code ==
          "spec.unknown_option");
}

TEST_CASE("annotation.remove takes the annotation out, or answers annotation.not_found") {
    const spec::Spec s = base();
    SpecEdit a = apply_spec_edit(s, "annotation.set", {{"panel", "loss"}, {"annotation", threshold("floor", 0.1)}});
    REQUIRE(!a.error);
    SpecEdit r = apply_spec_edit(a.spec, "annotation.remove", {{"panel", "loss"}, {"id", "floor"}});
    REQUIRE_MESSAGE(!r.error, r.error.message);
    CHECK(r.spec.panels[0].annotations.empty());
    CHECK(r.result == json{{"panel", "loss"}});
    CHECK(apply_spec_edit(a.spec, "annotation.remove", {{"panel", "loss"}, {"id", "gone"}}).error.code ==
          "annotation.not_found");
    CHECK(apply_spec_edit(a.spec, "annotation.remove", {{"panel", "nope"}, {"id", "floor"}}).error.code ==
          "panel.not_found");
}
