// The API file: its strict parse, its errors (which never hold the token) and
// where it is found.
#include <caliper_builder/mcp/api_file.h>

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace caliper_builder::mcp;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

const std::string kToken = "Zq9_abcdefghijklmnopqrstuvwxyz-0";

json good_file() {
    return {{"version", 1},
            {"pid", 4242},
            {"port", 7341},
            {"token", kToken},
            {"url", "http://127.0.0.1:7341/v0"},
            {"stream_url", "ws://127.0.0.1:7341/v3/stream"}};
}

fs::path tmp_dir(const std::string& name) {
    const fs::path p = fs::path(CALIPER_BUILDER_TEST_TMP) / "api-file" / name;
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

}  // namespace

TEST_CASE("a well-formed API file gives the endpoint") {
    const ApiFileRead r = parse_api_file(good_file().dump(), "api.json");
    REQUIRE(r.error.ok());
    CHECK(r.endpoint.port == 7341);
    CHECK(r.endpoint.token == kToken);
    CHECK(r.endpoint.pid == 4242);
    CHECK(r.endpoint.url == "http://127.0.0.1:7341/v0");
    CHECK(r.endpoint.stream_url == "ws://127.0.0.1:7341/v3/stream");

    json extra = good_file();
    extra["later_key"] = {{"a", 1}};
    extra.erase("pid");
    extra.erase("stream_url");
    const ApiFileRead r2 = parse_api_file(extra.dump(), "api.json");
    CHECK(r2.error.ok());
    CHECK(r2.endpoint.pid == 0);
}

TEST_CASE("a malformed or hostile API file is refused without echoing the token") {
    auto refused = [](const json& j) {
        const ApiFileRead r = parse_api_file(j.dump(), "api.json");
        CHECK(r.error.code == "mcp.bad_api_file");
        CHECK(r.error.message.find(kToken) == std::string::npos);
        CHECK(r.endpoint.token.empty());
        return r;
    };
    json j = good_file();
    j["version"] = 2;
    refused(j);
    j = good_file();
    j.erase("port");
    refused(j);
    for (const json& port : {json(0), json(65536), json(-1), json("7341"), json(7341.5)}) {
        j = good_file();
        j["port"] = port;
        refused(j);
    }
    j = good_file();
    j["url"] = "http://localhost:7341/v0";
    refused(j);
    j = good_file();
    j["url"] = "http://10.0.0.5:7341/v0";
    refused(j);
    j = good_file();
    j["url"] = "http://127.0.0.1:7342/v0";
    refused(j);
    j = good_file();
    j["token"] = "short";
    refused(j);
    j = good_file();
    j["token"] = kToken + "\r\nX-Evil: 1";
    refused(j);
    j = good_file();
    j["token"] = std::string(300, 'a');
    refused(j);
    j = good_file();
    j["pid"] = "4242";
    refused(j);
    refused(json::array({good_file()}));

    CHECK(parse_api_file("{not json " + kToken, "api.json").error.code == "mcp.bad_api_file");
    CHECK(parse_api_file(std::string(10000, '[') + std::string(10000, ']'), "api.json").error.code ==
          "mcp.bad_api_file");
}

TEST_CASE("reading the API file reports a missing, oversized or non-regular file") {
    const fs::path dir = tmp_dir("read");
    CHECK(read_api_file((dir / "absent.json").string()).error.code == "mcp.no_api_file");
    CHECK(read_api_file("").error.code == "mcp.no_api_file");

    {
        std::ofstream out(dir / "api.json");
        out << good_file().dump();
    }
    const ApiFileRead ok = read_api_file((dir / "api.json").string());
    CHECK(ok.error.ok());
    CHECK(ok.endpoint.port == 7341);

    {
        std::ofstream out(dir / "big.json");
        out << std::string(70 << 10, ' ') << good_file().dump();
    }
    CHECK(read_api_file((dir / "big.json").string()).error.code == "mcp.bad_api_file");

    fs::create_directories(dir / "folder.json");
    CHECK(read_api_file((dir / "folder.json").string()).error.code == "mcp.bad_api_file");

    {
        std::ofstream out(dir / "half.json");
        out << good_file().dump().substr(0, 30);
    }
    const ApiFileRead half = read_api_file((dir / "half.json").string());
    CHECK(half.error.code == "mcp.bad_api_file");
    CHECK(half.error.message.find(kToken.substr(0, 10)) == std::string::npos);
}

TEST_CASE("the default API file follows CALIPER_DATA_DIR") {
    const char* before = std::getenv("CALIPER_DATA_DIR");
    const std::string saved = before ? before : "";
    const fs::path dir = tmp_dir("data-dir");
#ifndef _WIN32
    setenv("CALIPER_DATA_DIR", dir.string().c_str(), 1);
    CHECK(default_api_file_path() == (dir / "api.json").string());
    setenv("CALIPER_DATA_DIR", "", 1);
    const std::string os_default = default_api_file_path();
    CHECK(os_default != (dir / "api.json").string());
    if (!os_default.empty()) CHECK(fs::path(os_default).filename() == "api.json");
    if (before) setenv("CALIPER_DATA_DIR", saved.c_str(), 1);
    else unsetenv("CALIPER_DATA_DIR");
#endif
}
