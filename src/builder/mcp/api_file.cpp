// The API file of Caliper API (api_file.h).
#include <caliper_builder/mcp/api_file.h>

#include "codec.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>

#if defined(__APPLE__)
#include <crt_externs.h>
#elif !defined(_WIN32)
#include <unistd.h>
#endif

namespace caliper_builder::mcp {

namespace {

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// A test or hidden run by the app's rule (src/app_paths.h, THE GUARD):
// CALIPER_HIDDEN=1, or a CALIPER_TEST_* variable that is set and not empty.
bool test_or_hidden_run() {
    if (env("CALIPER_HIDDEN") == "1") return true;
#if defined(_WIN32)
    char** e = _environ;
#elif defined(__APPLE__)
    char** e = *_NSGetEnviron();
#else
    char** e = environ;
#endif
    for (; e && *e; ++e) {
        const char* eq = std::strchr(*e, '=');
        if (std::strncmp(*e, "CALIPER_TEST_", 13) == 0 && eq && eq[1] != '\0') return true;
    }
    return false;
}

bool token_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

}  // namespace

std::string default_api_file_path() {
    namespace fs = std::filesystem;
    const std::string data_dir = env("CALIPER_DATA_DIR");
    if (!data_dir.empty()) return (fs::path(data_dir) / "api.json").string();
    // A test or hidden run never reads your own app's API file, so it never
    // drives the app you are using.
    if (test_or_hidden_run()) return {};
#if defined(_WIN32)
    const std::string appdata = env("APPDATA");
    if (appdata.empty()) return {};
    return (fs::path(appdata) / "Caliper" / "api.json").string();
#elif defined(__APPLE__)
    const std::string home = env("HOME");
    if (home.empty()) return {};
    return (fs::path(home) / "Library" / "Application Support" / "Caliper" / "api.json").string();
#else
    const std::string xdg = env("XDG_DATA_HOME");
    if (!xdg.empty()) return (fs::path(xdg) / "caliper" / "api.json").string();
    const std::string home = env("HOME");
    if (home.empty()) return {};
    return (fs::path(home) / ".local" / "share" / "caliper" / "api.json").string();
#endif
}

ApiFileRead parse_api_file(const std::string& text, const std::string& where) {
    ApiFileRead out;
    auto bad = [&](const std::string& why) {
        out.endpoint = ApiEndpoint{};
        out.error = {"mcp.bad_api_file", "The API file is not one Caliper writes: " + why + ".", where};
        return out;
    };
    if (!detail::json_depth_ok(text, 16)) return bad("it nests too deeply");
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded()) return bad("it is not JSON");
    if (!j.is_object()) return bad("it is not a JSON object");
    const auto version = j.find("version");
    if (version == j.end() || !version->is_number_integer() || version->get<std::int64_t>() != 1)
        return bad("its version is not 1");
    const auto port = j.find("port");
    if (port == j.end() || !port->is_number_integer()) return bad("it has no port");
    const std::int64_t p = port->get<std::int64_t>();
    if (p < 1 || p > 65535) return bad("its port is not between 1 and 65535");
    out.endpoint.port = int(p);
    const auto token = j.find("token");
    if (token == j.end() || !token->is_string()) return bad("it has no session token");
    const std::string& t = token->get_ref<const std::string&>();
    if (t.size() < 16 || t.size() > 256) return bad("its session token does not have 16 to 256 characters");
    for (char c : t)
        if (!token_char(c)) return bad("its session token holds a character other than A-Z, a-z, 0-9, '_' and '-'");
    out.endpoint.token = t;
    const auto url = j.find("url");
    if (url == j.end() || !url->is_string()) return bad("it has no url");
    const std::string expected = "http://127.0.0.1:" + std::to_string(p) + "/v0";
    if (url->get_ref<const std::string&>() != expected)
        return bad("its url is not " + expected + "; caliper-mcp talks only to 127.0.0.1 on the file's port");
    out.endpoint.url = expected;
    const auto pid = j.find("pid");
    if (pid != j.end()) {
        if (!pid->is_number_integer() || pid->get<std::int64_t>() < 0) return bad("its pid is not a process id");
        out.endpoint.pid = pid->get<std::int64_t>();
    }
    const auto stream = j.find("stream_url");
    if (stream != j.end()) {
        if (!stream->is_string() || stream->get_ref<const std::string&>().size() > 512)
            return bad("its stream_url is not a short string");
        out.endpoint.stream_url = stream->get<std::string>();
    }
    return out;
}

ApiFileRead read_api_file(const std::string& path) {
    ApiFileRead out;
    if (path.empty()) {
        out.error = {"mcp.no_api_file",
                     test_or_hidden_run()
                         ? "No API file path is known: a test or hidden run reads no API file of your own app; set "
                           "CALIPER_DATA_DIR or pass --api-file."
                         : "No API file path is known: set CALIPER_DATA_DIR or pass --api-file.",
                     ""};
        return out;
    }
    // The app may be writing the file at this moment; a file that does not
    // parse is read once more after a short pause.
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string text, why;
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            out.error = {"mcp.no_api_file",
                         "Caliper API is not running: there is no API file. Start the app with --api, or pass "
                         "--api-file with the file it writes.",
                         path};
            return out;
        }
        if (!detail::read_small_file(path, 64u << 10, &text, &why)) {
            out.error = {"mcp.bad_api_file", "The API file cannot be read: " + why + ".", path};
            return out;
        }
        out = parse_api_file(text, path);
        if (out.error.ok() || attempt == 1) return out;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return out;
}

}  // namespace caliper_builder::mcp
