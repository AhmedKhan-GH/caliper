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
        // The id becomes a data-folder name: path escapes and non-[a-z0-9.-]
        // characters are refused at parse time, before any loader sees them.
        {"[applet]\nid=\"../../escape\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n", "applet id"},
        {"[applet]\nid=\"a..b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n", "applet id"},
        {"[applet]\nid=\"Dev.Caliper.X\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n", "applet id"},
        {"[applet]\nid=\"a/b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n", "applet id"},
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

TEST_CASE("applet id validation") {
    CHECK(is_valid_applet_id("dev.caliper.hello"));
    CHECK(is_valid_applet_id("dev.caliper.embed-scope"));
    CHECK(is_valid_applet_id("a.b"));
    CHECK(is_valid_applet_id("x9"));
    CHECK_FALSE(is_valid_applet_id(""));
    CHECK_FALSE(is_valid_applet_id("."));
    CHECK_FALSE(is_valid_applet_id(".."));
    CHECK_FALSE(is_valid_applet_id("../escape"));
    CHECK_FALSE(is_valid_applet_id("a..b"));
    CHECK_FALSE(is_valid_applet_id(".lead"));
    CHECK_FALSE(is_valid_applet_id("trail."));
    CHECK_FALSE(is_valid_applet_id("Dev.Caliper"));     // uppercase
    CHECK_FALSE(is_valid_applet_id("dev.under_score"));
    CHECK_FALSE(is_valid_applet_id("a/b"));
    CHECK_FALSE(is_valid_applet_id("a\\b"));
    CHECK_FALSE(is_valid_applet_id("/abs"));
    CHECK_FALSE(is_valid_applet_id("a b"));
    CHECK_FALSE(is_valid_applet_id(std::string(129, 'a')));
    CHECK(is_valid_applet_id(std::string(128, 'a')));
}

// The versioning keys (docs/explanations/compatibility.md): [compat].torch,
// [build].sdk, [build].ui_stack and [data].schema, all optional.
static const char* kMinimal =
    "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=2\n";

TEST_CASE("manifest: versioning keys parse") {
    auto r = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=2\ntorch=\"2.5.1\"\n"
        "[build]\nsdk=\"0.7.0\"\nui_stack=\"0123456789abcdef\"\n"
        "[data]\nschema=3\n");
    REQUIRE(r.ok);
    CHECK(r.manifest.torch == "2.5.1");
    CHECK(r.manifest.sdk == "0.7.0");
    CHECK(r.manifest.ui_stack == "0123456789abcdef");
    CHECK(r.manifest.data_schema == 3);
}

TEST_CASE("manifest: versioning keys are absent by default") {
    auto r = parse_manifest_text(kMinimal);
    REQUIRE(r.ok);
    CHECK(r.manifest.torch.empty());
    CHECK(r.manifest.sdk.empty());
    CHECK(r.manifest.ui_stack.empty());
    CHECK(r.manifest.data_schema == 0);
}

TEST_CASE("manifest: malformed versioning keys are manifest errors") {
    struct Case { const char* tail; const char* needle; };
    const Case cases[] = {
        {"[data]\nschema=0\n", "data.schema"},
        {"[data]\nschema=-2\n", "data.schema"},
        {"[data]\nschema=\"2\"\n", "data.schema"},
        {"[data]\nschema=1.5\n", "data.schema"},
        {"[build]\nui_stack=\"\"\n", "build.ui_stack"},
        {"[build]\nui_stack=7\n", "build.ui_stack"},
    };
    for (auto& c : cases) {
        const std::string text = std::string(kMinimal) + c.tail;
        CAPTURE(text);
        auto r = parse_manifest_text(text);
        CHECK_FALSE(r.ok);
        CHECK(r.error.find(c.needle) != std::string::npos);
    }
    // [compat].torch sits in the [compat] table the minimal text opens last.
    for (const char* torch : {"torch=\"\"\n", "torch=2\n"}) {
        auto r = parse_manifest_text(std::string(kMinimal) + torch);
        CAPTURE(torch);
        CHECK_FALSE(r.ok);
        CHECK(r.error.find("compat.torch") != std::string::npos);
    }
}

TEST_CASE("manifest: [build].sdk is a record, never an error") {
    auto r = parse_manifest_text(std::string(kMinimal) + "[build]\nsdk=7\n");
    REQUIRE(r.ok);
    CHECK(r.manifest.sdk.empty());
}
