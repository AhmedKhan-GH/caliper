// The spec runtime resolves a panel again when the generation of its served
// data changes (SpecPanelSource::generation()), and only that panel: the
// refresh of a live panel, headless through SpecRuntime::update().
#include "spec_runtime/spec_runtime.h"

#include "spec_test_data.h"

#include <doctest/doctest.h>

#include <map>
#include <string>

using namespace caliper_host;
using namespace spec_test;

namespace {

// Text panels that resolve to nothing, with a generation per panel the test
// sets, counting the resolves.
struct CountingSource final : SpecPanelSource {
    std::map<std::string, int> resolves;
    std::map<std::string, uint64_t> generations;
    PanelData resolve(const spec::Spec&, std::string_view panel_id) override {
        const std::string id(panel_id);
        ++resolves[id];
        PanelData d;
        d.panel = id;
        d.generation = generations[id];
        return d;
    }
    uint64_t generation(const spec::Spec&, std::string_view panel_id) override {
        return generations[std::string(panel_id)];
    }
};

}  // namespace

TEST_CASE("spec runtime: a panel whose served generation changes is resolved again, and only it") {
    const spec::Spec s = parse(
        "caliper_spec = 0\n[applet]\nid = \"t\"\n"
        "[[panel]]\nid = \"a\"\nprimitive = \"text\"\noptions = { text = \"one\" }\n"
        "[[panel]]\nid = \"b\"\nprimitive = \"text\"\noptions = { text = \"two\" }\n");
    SpecTextures textures;
    SpecRuntime runtime(textures);
    CountingSource source;
    runtime.set_source(&source);
    runtime.set_spec(s);
    runtime.update();
    CHECK(source.resolves["a"] == 1);
    CHECK(source.resolves["b"] == 1);
    // Nothing new: nothing is resolved.
    runtime.update();
    runtime.update();
    CHECK(source.resolves["a"] == 1);
    CHECK(source.resolves["b"] == 1);
    // A new snapshot behind a: a alone.
    source.generations["a"] = 7;
    runtime.update();
    CHECK(source.resolves["a"] == 2);
    CHECK(source.resolves["b"] == 1);
    runtime.update();
    CHECK(source.resolves["a"] == 2);
}

namespace {

// A series panel whose next resolve the test sets: data, or an error.
struct ScriptedSource final : SpecPanelSource {
    PanelData next;
    uint64_t gen = 1;
    PanelData resolve(const spec::Spec&, std::string_view panel_id) override {
        PanelData d = next;
        d.panel = std::string(panel_id);
        d.primitive = spec::Primitive::series;
        d.generation = gen;
        return d;
    }
    uint64_t generation(const spec::Spec&, std::string_view) override { return gen; }
    void put(PanelData d) {
        next = std::move(d);
        ++gen;
    }
};

PanelData series(std::vector<double> y) {
    PanelData d;
    d.slots = {column("y", "loss.value", json(y))};
    return d;
}

PanelData failed(const std::string& code) {
    PanelData d;
    d.error = caliper_builder::Error(code, "steps reads the state n, which is -3 now.", "source recent, transform[0]");
    return d;
}

}  // namespace

// A reference whose value is outside the range of its place stops the chain
// with resolve.out_of_range: the panel keeps its last good data and shows the
// error in its header, and a good value clears it. Without last good data, the
// panel shows the error state.
TEST_CASE("spec runtime: a value out of range keeps the panel's last good data with the error in its header") {
    const spec::Spec s = parse("caliper_spec = 0\n[applet]\nid = \"t\"\n"
                               "[[panel]]\nid = \"a\"\nprimitive = \"series\"\nbind = { y = \"loss.value\" }\n");
    SpecTextures textures;
    SpecRuntime runtime(textures);
    ScriptedSource source;
    runtime.set_source(&source);

    source.put(failed("resolve.out_of_range"));
    runtime.set_spec(s);
    runtime.update();
    const PreparedPanel* p = runtime.prepared("a");
    REQUIRE(p);
    CHECK(p->state.label == "Value out of range");
    CHECK(p->notice.ready());

    source.put(series({1, 2, 3}));
    runtime.update();
    p = runtime.prepared("a");
    REQUIRE(p);
    REQUIRE(p->state.ready());
    REQUIRE(p->series.lines.size() == 1);
    CHECK(p->series.lines[0].count == 3);
    CHECK(p->notice.ready());

    source.put(failed("resolve.out_of_range"));
    runtime.update();
    p = runtime.prepared("a");
    REQUIRE(p);
    CHECK(p->state.ready());
    REQUIRE(p->series.lines.size() == 1);
    CHECK(p->series.lines[0].count == 3);   // the last good data
    CHECK(p->notice.label == "Value out of range");
    CHECK(p->notice.detail == "source recent, transform[0]");

    source.put(series({4, 5}));
    runtime.update();
    p = runtime.prepared("a");
    REQUIRE(p);
    CHECK(p->series.lines[0].count == 2);
    CHECK(p->notice.ready());

    // Another error is no range error: the panel shows it and keeps nothing.
    source.put(failed("resolve.unavailable"));
    runtime.update();
    p = runtime.prepared("a");
    REQUIRE(p);
    CHECK_FALSE(p->state.ready());
    CHECK(p->notice.ready());
}
