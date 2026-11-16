// The resources: listing, reading, and the confinement of the docs and SDK
// folders against path tricks and symbolic links.
#include <caliper_builder/mcp/resources.h>

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <string>

using namespace caliper_builder::mcp;
namespace fs = std::filesystem;

namespace {

void write(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out << text;
}

struct Fixture {
    fs::path root;
    fs::path docs;
    fs::path sdk;
    fs::path outside;

    Fixture() {
        root = fs::path(CALIPER_BUILDER_TEST_TMP) / "resources";
        fs::remove_all(root);
        docs = root / "docs";
        sdk = root / "sdk";
        outside = root / "outside";
        write(docs / "index.md", "# Caliper\n");
        write(docs / "references" / "api.md", "# Caliper API\n");
        write(docs / "plans" / "deep" / "page.md", "# Deep\n");
        write(docs / "notes.txt", "not served\n");
        write(docs / ".hidden.md", "hidden\n");
        write(docs / ".git" / "config.md", "hidden folder\n");
        write(docs / "big.md", std::string(5u << 20, 'x'));
        write(docs / "space name.md", "a name the grammar refuses\n");
        write(sdk / "include" / "caliper" / "sdk.h", "#pragma once\n");
        write(sdk / "include" / "caliper" / "detail.hpp", "#pragma once\n");
        write(sdk / "lib" / "cmake" / "caliper-sdk-config.cmake", "# not served\n");
        write(outside / "secret.md", "a secret outside the folder\n");
        write(outside / "secret.h", "a secret header\n");
#ifndef _WIN32
        fs::create_symlink(outside / "secret.md", docs / "escape.md");
        fs::create_directory_symlink(outside, docs / "linked");
        fs::create_symlink(docs / "index.md", docs / "alias.md");
        fs::create_symlink(docs / "notes.txt", docs / "notes.md");
        fs::create_symlink(outside / "secret.h", sdk / "include" / "escape.h");
#endif
    }
};

std::set<std::string> uris(const std::vector<ResourceEntry>& entries) {
    std::set<std::string> out;
    for (const auto& e : entries) out.insert(e.uri);
    return out;
}

}  // namespace

TEST_CASE("checked_resource_path accepts plain relative paths only") {
    CHECK(checked_resource_path("index.md") == "index.md");
    CHECK(checked_resource_path("references/api.md") == "references/api.md");
    CHECK(checked_resource_path("references%2Fapi.md") == "references/api.md");
    CHECK(checked_resource_path("a_b-c.1/x.md") == "a_b-c.1/x.md");
    for (const char* bad : {"", "/etc/passwd", "../outside/secret.md", "a/../b.md", "a/./b.md", "%2e%2e/secret.md",
                            "%2E%2E%2Fsecret.md", "a//b.md", "a/", ".hidden.md", "a/.git/x.md", "a\\b.md",
                            "a%5cb.md", "a%00.md", "a%2", "a%zz.md", "space name.md", "a%20b.md", "C:/x.md",
                            "caf\xc3\xa9.md"})
        CHECK_MESSAGE(checked_resource_path(bad).empty(), bad);
    CHECK(checked_resource_path(std::string(2000, 'a')).empty());
    std::string deep;
    for (int i = 0; i < 40; ++i) deep += "d/";
    CHECK(checked_resource_path(deep + "x.md").empty());
}

TEST_CASE("the store lists the served files and nothing else") {
    Fixture f;
    ResourceStore store;
    ResourceOptions o;
    o.docs_dir = f.docs.string();
    o.sdk_dir = f.sdk.string();
    o.spec_schema_text = "{\"type\": \"object\"}";
    REQUIRE(store.open(o).ok());
    bool truncated = true;
    const auto entries = store.list(&truncated);
    CHECK_FALSE(truncated);
    std::set<std::string> want = {"caliper://docs/index.md", "caliper://docs/references/api.md",
                                  "caliper://docs/plans/deep/page.md", "caliper://sdk/include/caliper/sdk.h",
                                  "caliper://sdk/include/caliper/detail.hpp", kSpecSchemaUri};
#ifndef _WIN32
    want.insert("caliper://docs/alias.md");   // a link that stays inside
#endif
    CHECK(uris(entries) == want);
    for (std::size_t i = 1; i < entries.size(); ++i) CHECK(entries[i - 1].uri < entries[i].uri);
    for (const auto& e : entries) {
        if (e.uri == "caliper://docs/index.md") {
            CHECK(e.mime_type == "text/markdown");
            CHECK(e.name == "index.md");
            CHECK(e.size == 10);
            CHECK(e.to_json()["uri"] == e.uri);
        }
        if (e.uri == "caliper://sdk/include/caliper/sdk.h") CHECK(e.mime_type == "text/x-c");
        if (e.uri == kSpecSchemaUri) CHECK(e.mime_type == "application/schema+json");
    }
    CHECK(store.templates().size() == 2);
}

TEST_CASE("the store reads served files and refuses every escape") {
    Fixture f;
    ResourceStore store;
    ResourceOptions o;
    o.docs_dir = f.docs.string();
    o.sdk_dir = f.sdk.string();
    o.spec_schema_text = "{\"type\": \"object\"}";
    REQUIRE(store.open(o).ok());

    ResourceContent c = store.read("caliper://docs/references/api.md");
    REQUIRE(c.error.ok());
    CHECK(c.text == "# Caliper API\n");
    CHECK(c.mime_type == "text/markdown");
    CHECK(store.read("caliper://docs/references%2Fapi.md").text == "# Caliper API\n");
    CHECK(store.read("caliper://sdk/include/caliper/sdk.h").text == "#pragma once\n");
    CHECK(store.read(kSpecSchemaUri).text == "{\"type\": \"object\"}");

    for (const char* bad : {"caliper://docs/../outside/secret.md", "caliper://docs/%2e%2e/outside/secret.md",
                            "caliper://docs/escape.md", "caliper://docs/linked/secret.md", "caliper://docs/notes.txt",
                            "caliper://docs/notes.md", "caliper://docs/.hidden.md", "caliper://docs/.git/config.md",
                            "caliper://docs/missing.md", "caliper://docs/references", "caliper://docs/",
                            "caliper://sdk/include/escape.h", "caliper://sdk/lib/cmake/caliper-sdk-config.cmake",
                            "caliper://docs//etc/passwd", "file:///etc/passwd", "caliper://other/x.md",
                            "caliper://schema/other.json", "caliper://docs/space%20name.md"}) {
        const ResourceContent r = store.read(bad);
        CHECK_MESSAGE(r.error.code == "mcp.resource_not_found", bad);
        CHECK(r.text.empty());
    }
    const ResourceContent big = store.read("caliper://docs/big.md");
    CHECK(big.error.code == "mcp.resource_unreadable");
}

TEST_CASE("the store without folders serves the schema only, and a bad folder is refused") {
    ResourceStore store;
    ResourceOptions o;
    o.spec_schema_text = "{}";
    REQUIRE(store.open(o).ok());
    CHECK(store.list().size() == 1);
    CHECK(store.templates().empty());
    CHECK(store.read("caliper://docs/index.md").error.code == "mcp.resource_not_found");

    ResourceStore bad;
    ResourceOptions b;
    b.docs_dir = (fs::path(CALIPER_BUILDER_TEST_TMP) / "no-such-folder").string();
    CHECK(bad.open(b).code == "mcp.bad_resource_dir");
    Fixture f;
    b.docs_dir = (f.docs / "index.md").string();
    CHECK(bad.open(b).code == "mcp.bad_resource_dir");
}

TEST_CASE("listing stops at its bounds") {
    const fs::path root = fs::path(CALIPER_BUILDER_TEST_TMP) / "resources-bounds";
    fs::remove_all(root);
    for (int i = 0; i < 30; ++i) write(root / ("p" + std::to_string(i) + ".md"), "x");
    fs::path deep = root;
    for (int i = 0; i < 20; ++i) deep /= "d";
    write(deep / "deep.md", "x");
    ResourceStore store;
    ResourceOptions o;
    o.docs_dir = root.string();
    o.max_files = 10;
    REQUIRE(store.open(o).ok());
    bool truncated = false;
    const auto entries = store.list(&truncated);
    CHECK(truncated);
    CHECK(entries.size() == 10);

    o.max_files = 5000;
    REQUIRE(store.open(o).ok());
    const auto all = store.list(&truncated);
    CHECK_FALSE(truncated);
    CHECK(all.size() == 30);   // deep.md is below the depth bound
}
