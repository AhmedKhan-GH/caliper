// ArtifactStore — the content-addressed blob store behind caliper.artifacts.v1
// (PLATFORM.md §7.8, §16 contract): put round-trips a digest, identical bytes
// dedup to one file, a name resolves to its newest digest, unknowns are inert.
#include <doctest/doctest.h>

#include "artifact_store.h"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using caliper_host::ArtifactStore;

namespace {

struct TempRoot {
    fs::path dir;
    TempRoot() {
        dir = fs::temp_directory_path() /
              ("caliper-artifacts-test-" + std::to_string(::getpid()));
        fs::remove_all(dir);
        fs::create_directories(dir);
    }
    ~TempRoot() { std::error_code ec; fs::remove_all(dir, ec); }
};

size_t files_in(const fs::path& p) {
    size_t n = 0;
    if (fs::exists(p))
        for (auto& e : fs::directory_iterator(p))
            if (e.is_regular_file()) n++;
    return n;
}

}  // namespace

TEST_CASE("artifact_store: put round-trips through exists and path_of") {
    TempRoot t;
    ArtifactStore store;
    REQUIRE(store.open(t.dir.string()));

    const char payload[] = "not actually a checkpoint";
    char digest[65] = {};
    REQUIRE(store.put("model-a", payload, sizeof(payload), 7, digest));

    CHECK(std::strlen(digest) == 64);
    CHECK(digest[64] == '\0');
    // hex only
    for (int i = 0; i < 64; i++)
        CHECK(std::isxdigit(static_cast<unsigned char>(digest[i])));

    CHECK(store.exists(digest));
    CHECK(store.exists("model-a"));

    std::string path = store.path_of(digest);
    REQUIRE_FALSE(path.empty());
    CHECK(fs::exists(path));
    // The stored file's bytes are the payload, verbatim.
    std::ifstream in(path, std::ios::binary);
    std::string back((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    CHECK(back == std::string(payload, sizeof(payload)));
    // path_of by name resolves to the same file.
    CHECK(store.path_of("model-a") == path);
}

TEST_CASE("artifact_store: identical bytes dedup to one file, same digest") {
    TempRoot t;
    ArtifactStore store;
    REQUIRE(store.open(t.dir.string()));

    const char payload[] = "same bytes twice";
    char d1[65] = {}, d2[65] = {};
    REQUIRE(store.put("first", payload, sizeof(payload), 1, d1));
    REQUIRE(store.put("second", payload, sizeof(payload), 2, d2));

    CHECK(std::string(d1) == std::string(d2));
    CHECK(files_in(t.dir / "artifacts") == 1);
    // Both names resolve (to the same blob).
    CHECK(store.exists("first"));
    CHECK(store.exists("second"));
}

TEST_CASE("artifact_store: a reused name resolves to the newest digest") {
    TempRoot t;
    ArtifactStore store;
    REQUIRE(store.open(t.dir.string()));

    const char v1[] = "weights v1";
    const char v2[] = "weights v2 (better)";
    char d1[65] = {}, d2[65] = {};
    REQUIRE(store.put("model", v1, sizeof(v1), 1, d1));
    REQUIRE(store.put("model", v2, sizeof(v2), 1, d2));
    REQUIRE(std::string(d1) != std::string(d2));

    // Name -> newest row's digest; both digests remain addressable.
    CHECK(store.path_of("model") == store.path_of(d2));
    CHECK(store.exists(d1));
    CHECK(files_in(t.dir / "artifacts") == 2);
}

TEST_CASE("artifact_store: unknown digests and names are inert") {
    TempRoot t;
    ArtifactStore store;
    REQUIRE(store.open(t.dir.string()));

    CHECK_FALSE(store.exists("no-such-name"));
    CHECK_FALSE(store.exists(
        "0000000000000000000000000000000000000000000000000000000000000000"));
    CHECK(store.path_of("no-such-name").empty());
}

TEST_CASE("artifact_store: run lineage is queryable") {
    TempRoot t;
    ArtifactStore store;
    REQUIRE(store.open(t.dir.string()));

    char d1[65] = {}, d2[65] = {}, d3[65] = {};
    const char a[] = "aaa", b[] = "bbb", c[] = "ccc";
    REQUIRE(store.put("ckpt-1", a, sizeof(a), 42, d1));
    REQUIRE(store.put("ckpt-2", b, sizeof(b), 42, d2));
    REQUIRE(store.put("other", c, sizeof(c), 99, d3));

    auto of42 = store.by_run(42);
    REQUIRE(of42.size() == 2);
    CHECK(((of42[0] == d1 && of42[1] == d2) ||
           (of42[0] == d2 && of42[1] == d1)));
    CHECK(store.by_run(7).empty());
}

TEST_CASE("artifact_store: unopened store fails put, stays inert") {
    ArtifactStore store;  // never opened
    char digest[65] = {};
    CHECK_FALSE(store.put("x", "y", 1, 0, digest));
    CHECK_FALSE(store.exists("x"));
    CHECK(store.path_of("x").empty());
}
