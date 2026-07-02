#include <doctest/doctest.h>
#include "applet_manifest.h"
using namespace caliper_host;

static const char* kGolden = R"([applet]
id = "dev.ahmed.circuitnet"
name = "CircuitNet 3.0"
version = "1.0.0"
summary = "Gate-level circuit explorer"
tag = "EDA"

[compat]
abi_epoch = 2
min_host = "0.6.0"

[services]
required = ["caliper.ui.v1"]
optional = ["caliper.log.v1"]
)";

TEST_CASE("manifest: golden parses fully") {
    auto r = parse_manifest_text(kGolden);
    REQUIRE(r.ok);
    CHECK(r.manifest.id == "dev.ahmed.circuitnet");
    CHECK(r.manifest.name == "CircuitNet 3.0");
    CHECK(r.manifest.version == "1.0.0");
    CHECK(r.manifest.summary == "Gate-level circuit explorer");
    CHECK(r.manifest.tag == "EDA");
    CHECK(r.manifest.abi_epoch == 2);
    CHECK(r.manifest.min_host == "0.6.0");
    REQUIRE(r.manifest.required_services.size() == 1);
    CHECK(r.manifest.required_services[0] == "caliper.ui.v1");
    REQUIRE(r.manifest.optional_services.size() == 1);
    CHECK(r.manifest.optional_services[0] == "caliper.log.v1");
}

TEST_CASE("manifest: minimal — only id/name/version/epoch required") {
    auto r = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=2\n");
    REQUIRE(r.ok);
    CHECK(r.manifest.min_host.empty());
    CHECK(r.manifest.required_services.empty());
    CHECK(r.manifest.summary.empty());
}

TEST_CASE("manifest: adversarial inputs refuse with a reason") {
    struct Case { const char* toml; const char* needle; };
    const Case cases[] = {
        {"", "missing"},                                              // empty
        {"not toml {{{", "parse"},                                    // syntax
        {"[applet]\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n", "id"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\n[compat]\nabi_epoch=2\n", "version"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"1.0\"\n[compat]\nabi_epoch=2\n", "semver"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n", "abi_epoch"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=\"two\"\n", "abi_epoch"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\nmin_host=\"soon\"\n", "semver"},
    };
    for (auto& c : cases) {
        auto r = parse_manifest_text(c.toml);
        CAPTURE(c.toml);
        CHECK_FALSE(r.ok);
        CHECK(r.error.find(c.needle) != std::string::npos);
    }
}

TEST_CASE("manifest: unknown tables/keys are ignored (forward compat)") {
    auto r = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\nauthors=[\"x\"]\n"
        "[compat]\nabi_epoch=2\n[future]\nx=1\n");
    CHECK(r.ok);
}

TEST_CASE("semver validation") {
    CHECK(is_valid_semver("0.6.0"));
    CHECK(is_valid_semver("10.20.30"));
    CHECK_FALSE(is_valid_semver("1.0"));
    CHECK_FALSE(is_valid_semver("v1.0.0"));
    CHECK_FALSE(is_valid_semver(""));
}
