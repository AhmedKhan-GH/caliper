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
abi_epoch = 3
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
    CHECK(r.manifest.abi_epoch == 3);
    CHECK(r.manifest.min_host == "0.6.0");
    REQUIRE(r.manifest.required_services.size() == 1);
    CHECK(r.manifest.required_services[0] == "caliper.ui.v1");
    REQUIRE(r.manifest.optional_services.size() == 1);
    CHECK(r.manifest.optional_services[0] == "caliper.log.v1");
}

TEST_CASE("manifest: minimal — only id/name/version/epoch required") {
    auto r = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=3\n");
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
        {"[applet]\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=3\n", "id"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\n[compat]\nabi_epoch=3\n", "version"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"1.0\"\n[compat]\nabi_epoch=3\n", "semver"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n", "abi_epoch"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=\"two\"\n", "abi_epoch"},
        {"[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=3\nmin_host=\"soon\"\n", "semver"},
        // The id becomes a data-folder name: path escapes and non-[a-z0-9.-]
        // characters are refused at parse time, before any loader sees them.
        {"[applet]\nid=\"../../escape\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=3\n", "applet id"},
        {"[applet]\nid=\"a..b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=3\n", "applet id"},
        {"[applet]\nid=\"Dev.Caliper.X\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=3\n", "applet id"},
        {"[applet]\nid=\"a/b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=3\n", "applet id"},
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
        "[compat]\nabi_epoch=3\n[future]\nx=1\n");
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
    CHECK(is_valid_applet_id("dev.caliper.hello-world"));
    CHECK(is_valid_applet_id("dev.caliper.mnist"));
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

// The versioning keys (docs/development/compatibility-and-epochs.md): [compat].torch,
// [build].sdk, [build].ui_stack and [data].schema, all optional.
static const char* kMinimal =
    "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n[compat]\nabi_epoch=3\n";

TEST_CASE("manifest: versioning keys parse") {
    auto r = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n"
        "[compat]\nabi_epoch=3\ntorch=\"2.5.1\"\n"
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

// ---- spec applets (kind = "spec") --------------------------------------------
// docs/applets/manifest.md, "Spec applets": a manifest and a spec, no
// library; the manifest needs no [compat] and refuses what only a library can
// honour.
static const char* kSpecManifest =
    "[applet]\nid=\"local.demo\"\nname=\"Demo\"\nversion=\"0.1.0\"\n"
    "kind=\"spec\"\nspec=\"demo.spec.toml\"\n";

TEST_CASE("manifest: a spec applet needs no [compat] and names its spec") {
    auto r = parse_manifest_text(kSpecManifest);
    REQUIRE(r.ok);
    CHECK(r.manifest.kind == AppletKind::Spec);
    CHECK(r.manifest.spec == "demo.spec.toml");
    CHECK(r.manifest.abi_epoch == 0);
    CHECK(r.manifest.min_host.empty());

    // summary, tag and [compat].min_host keep their meaning.
    auto full = parse_manifest_text(std::string(kSpecManifest) +
                                    "summary=\"One line.\"\ntag=\"Run\"\n"
                                    "[compat]\nmin_host=\"0.6.0\"\n[future]\nx=1\n");
    REQUIRE(full.ok);
    CHECK(full.manifest.summary == "One line.");
    CHECK(full.manifest.tag == "Run");
    CHECK(full.manifest.min_host == "0.6.0");

    // What applet.save writes (design/builder-session.md) parses as it is.
    auto saved = parse_manifest_text(
        "[applet]\nid = \"local.my-view\"\nname = \"My view\"\nversion = \"0.1.0\"\n"
        "summary = \"Loss over steps.\"\nkind = \"spec\"\nspec = \"my_view.spec.toml\"\n");
    REQUIRE(saved.ok);
    CHECK(saved.manifest.spec == "my_view.spec.toml");
}

TEST_CASE("manifest: an absent or library kind is a compiled applet as before") {
    auto absent = parse_manifest_text(kMinimal);
    REQUIRE(absent.ok);
    CHECK(absent.manifest.kind == AppletKind::Library);
    auto library = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\nkind=\"library\"\n"
        "[compat]\nabi_epoch=3\n");
    REQUIRE(library.ok);
    CHECK(library.manifest.kind == AppletKind::Library);
    // A library still needs its epoch; spec is an unknown key to it, ignored.
    auto no_epoch = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\nkind=\"library\"\n");
    CHECK_FALSE(no_epoch.ok);
    CHECK(no_epoch.error.find("abi_epoch") != std::string::npos);
    auto stray = parse_manifest_text(
        "[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\nspec=\"x.spec.toml\"\n"
        "[compat]\nabi_epoch=3\n");
    REQUIRE(stray.ok);
    CHECK(stray.manifest.kind == AppletKind::Library);
    CHECK(stray.manifest.spec.empty());
}

TEST_CASE("manifest: each refusal of a spec applet's manifest, keeping its kind") {
    const std::string head =
        "[applet]\nid=\"local.demo\"\nname=\"Demo\"\nversion=\"0.1.0\"\nkind=\"spec\"\n";
    const std::string with_spec = head + "spec=\"demo.spec.toml\"\n";
    struct Case { std::string toml; const char* needle; };
    const Case cases[] = {
        {head, "manifest missing applet.spec"},
        {head + "spec=\"\"\n", "applet.spec is not a plain file name"},
        {head + "spec=\"../demo.spec.toml\"\n", "applet.spec is not a plain file name"},
        {head + "spec=\"sub/demo.spec.toml\"\n", "applet.spec is not a plain file name"},
        {head + "spec=\"sub\\\\demo.spec.toml\"\n", "applet.spec is not a plain file name"},
        {head + "spec=\"/etc/demo.spec.toml\"\n", "applet.spec is not a plain file name"},
        {head + "spec=\"C:demo.spec.toml\"\n", "applet.spec is not a plain file name"},
        {head + "spec=\"..\"\n", "applet.spec is not a plain file name"},
        {head + "spec=\"a\\u0001b\"\n", "applet.spec is not a plain file name"},
        {head + "spec=7\n", "applet.spec is not a plain file name"},
        {with_spec + "[compat]\nabi_epoch=3\n", "compat.abi_epoch does not apply to a spec applet"},
        {with_spec + "[compat]\ntorch=\"2.5.1\"\n", "compat.torch does not apply to a spec applet"},
        {with_spec + "[compat]\nmin_host=\"soon\"\n", "compat.min_host is not strict semver"},
        {"compat=3\n" + with_spec, "[compat] is not a table"},
        {with_spec + "[build]\nsdk=\"0.7.0\"\n", "[build] does not apply to a spec applet"},
        {with_spec + "[services]\nrequired=[\"caliper.ui.v1\"]\n", "[services] does not apply"},
        {with_spec + "[data]\nschema=1\n", "[data] does not apply to a spec applet"},
        // The identity is checked as for any applet, before the kind's keys.
        {"[applet]\nid=\"local.my_view\"\nname=\"A\"\nversion=\"0.1.0\"\nkind=\"spec\"\n"
         "spec=\"a.spec.toml\"\n", "applet id"},
        {"[applet]\nid=\"local.a\"\nversion=\"0.1.0\"\nkind=\"spec\"\nspec=\"a.spec.toml\"\n",
         "applet.name"},
    };
    for (const auto& c : cases) {
        CAPTURE(c.toml);
        auto r = parse_manifest_text(c.toml);
        CHECK_FALSE(r.ok);
        CHECK(r.error.find(c.needle) != std::string::npos);
        CHECK(r.manifest.kind == AppletKind::Spec);
    }
}

TEST_CASE("manifest: an unknown kind is a manifest error") {
    for (const char* kind : {"kind=\"wasm\"\n", "kind=\"Spec\"\n", "kind=1\n"}) {
        CAPTURE(kind);
        auto r = parse_manifest_text(
            std::string("[applet]\nid=\"a.b\"\nname=\"A\"\nversion=\"0.1.0\"\n") + kind +
            "[compat]\nabi_epoch=3\n");
        CHECK_FALSE(r.ok);
        CHECK(r.error.find("applet.kind") != std::string::npos);
        CHECK(r.manifest.kind == AppletKind::Library);
    }
}

TEST_CASE("plain file names") {
    CHECK(is_plain_file_name("demo.spec.toml"));
    CHECK(is_plain_file_name("my_view.spec.toml"));
    CHECK(is_plain_file_name(".hidden"));
    CHECK(is_plain_file_name("..x"));
    CHECK(is_plain_file_name(std::string(255, 'a')));
    CHECK_FALSE(is_plain_file_name(std::string(256, 'a')));
    for (const char* bad : {"", ".", "..", "a/b", "a\\b", "/abs", "c:x", "a\nb", "a\x7f"}) {
        CAPTURE(bad);
        CHECK_FALSE(is_plain_file_name(bad));
    }
    CHECK(std::string(applet_kind_word(AppletKind::Spec)) == "spec");
    CHECK(std::string(applet_kind_word(AppletKind::Library)) == "library");
}
