// The manifest parser and the loader as a build without CALIPER_WITH_BUILDER
// compiles them: this test's executable compiles their sources without
// CALIPER_HOST_SPEC_RUNTIME and links no builder library
// (tests/CMakeLists.txt), so it runs in every build. The manifest's kind is
// understood, and every kind = "spec" instrument is listed as refused because this build
// has no spec runtime; nothing is opened for it.
#include <doctest/doctest.h>
#include "instrument_loader.h"
#include "instrument_manifest.h"
#include "spec_instrument_fixture.h"
#include "test_temp.h"
#include <filesystem>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using namespace caliper_host;

namespace {
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(caliper_test::unique_temp_path("caliper-nospec-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};
} // namespace

TEST_CASE("no spec runtime: the kind is parsed and every kind = \"spec\" instrument is refused") {
    CHECK_FALSE(spec_runtime_available());
    CHECK(spec_format_version() == -1);

    auto parsed = parse_manifest_text(
        "[instrument]\nid=\"local.demo\"\nname=\"Demo\"\nversion=\"0.1.0\"\n"
        "kind=\"spec\"\nspec=\"demo.spec.toml\"\n");
    REQUIRE(parsed.ok);
    CHECK(parsed.manifest.kind == InstrumentKind::Spec);

    TempDir t("refused");
    caliper_test::write_spec_instrument(t.path / "ws" / "instruments" / "demo", "demo", "local.demo",
                                        "0.1.0");
    // A broken kind = "spec" instrument keeps its own reason: the file checks come first.
    caliper_test::write_spec_instrument(t.path / "ws" / "instruments" / "gone", "gone", "local.gone",
                                        "0.1.0", nullptr);
    InstrumentLoader L(HostCaps{4, "0.6.0", {}}, (t.path / "data").string());
    L.set_workspace((t.path / "ws").string());
    L.scan(instrument_search_dirs({}, nullptr, "", "", (t.path / "ws").string()));
    REQUIRE(L.count() == 2);
    int demo = -1, gone = -1;
    for (int i = 0; i < L.count(); i++)
        (L.at(i).manifest.id == "local.demo" ? demo : gone) = i;
    REQUIRE(demo >= 0);
    REQUIRE(gone >= 0);

    const InstrumentEntry& a = L.at(demo);
    CHECK(a.manifest.kind == InstrumentKind::Spec);
    CHECK(a.status == InstrumentStatus::Refused);
    CHECK(a.status_text == no_spec_runtime_reason());
    CHECK(a.status_text == "Requires the spec runtime, which this build of Caliper does not have.");
    CHECK(L.at(gone).status == InstrumentStatus::Failed);
    CHECK(L.at(gone).status_text ==
          "spec file gone.spec.toml not found next to gone.caliper.toml");

    std::vector<SpecDiagnostic> found(1);
    CHECK_FALSE(L.check_spec(demo, &found));
    CHECK(found.empty());
    CHECK_FALSE(L.probe(demo));
    CHECK_FALSE(L.launch(demo, CaliperHost{}));
    CHECK(L.at(demo).status == InstrumentStatus::Refused);
    CHECK(L.at(demo).status_text == no_spec_runtime_reason());
    CHECK(L.at(demo).handle == nullptr);
    CHECK_FALSE(fs::exists(t.path / "data"));
}
