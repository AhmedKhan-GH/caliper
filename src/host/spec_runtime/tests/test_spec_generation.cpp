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
