// The acceptance case "plot attention entropy per layer while it trains" in
// the hidden app (the hidden-app pattern of docs/development/testing.md).
// caliper_attention_producer (tests/live_attention_producer.cpp) stands in for
// a training script: a hub of its own and a producer that streams a model's
// attention as it sharpens. The app attaches to it with --url, and the test,
// through Caliper API's command endpoints alone, builds a spec over the live
// sources, opens it on the spec page, and checks that the frame and the
// heatmap panel change from one capture to the next while the stream runs,
// that the catalog shows the live sources with their shapes and rates, and
// that launching another instrument keeps the stream attached. POSIX with Metal
// only; without Metal the case is skipped.
//
// The other cases drive the hidden app's landing screen and live page through
// CALIPER_TEST_SPEC_UI beside one or two producers: F4 listing a stream from
// its stream file, the attachment kept across Home and the spec page until
// ⌘D Detach, a second stream replacing the first, a stream that ends while
// the landing shows, quitting while attached, and files dropped on the live
// page. The "app save" cases drive "Save changes to <name>?" the same way:
// F3 New dashboard, opening another dashboard from the landing, and closing
// the window, each over unsaved changes.
//
// CALIPER_TEST_SHOTS=<folder> saves the two captures there as
// live-entropy-a.png and live-entropy-b.png, and CALIPER_TEST_WINDOW=<W>x<H>
// sets the window's size in points: the capture script uses both.
#include <doctest/doctest.h>

#include "test_proc.h"
#include "test_temp.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <future>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;
using namespace std::chrono_literals;

namespace {

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name) : path(caliper_test::unique_temp_path("caliper-live-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str(const std::string& sub) const { return (path / sub).string(); }
};

struct Http {
    int status = 0;
    std::string body;
};

// One request to 127.0.0.1:<port> on a connection of its own.
Http http(uint16_t port, const std::string& method, const std::string& target, const std::string& token,
          const std::string& body = {}) {
    Http out;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    timeval tv{60, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    std::string req = method + " " + target + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                      "\r\nAuthorization: Bearer " + token + "\r\nConnection: close\r\n";
    if (method == "POST")
        req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n" + body;
    for (size_t sent = 0; sent < req.size();) {
        const ssize_t n = ::send(fd, req.data() + sent, req.size() - sent, 0);
        REQUIRE(n > 0);
        sent += (size_t)n;
    }
    std::string raw;
    char buf[65536];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        raw.append(buf, (size_t)n);
    }
    ::close(fd);
    const size_t head_end = raw.find("\r\n\r\n");
    REQUIRE_MESSAGE(head_end != std::string::npos, raw);
    out.status = std::atoi(raw.c_str() + raw.find(' ') + 1);
    out.body = raw.substr(head_end + 4);
    return out;
}

json command(uint16_t port, const std::string& token, const std::string& name, const json& args = json::object()) {
    const Http r = http(port, "POST", "/v0/commands/" + name, token, args.dump());
    REQUIRE_MESSAGE(r.status == 200, name << ": " << r.status << " " << r.body);
    const json env = json::parse(r.body, nullptr, false);
    REQUIRE(env.is_object());
    return env;
}

json ok(uint16_t port, const std::string& token, const std::string& name, const json& args = json::object()) {
    const json env = command(port, token, name, args);
    REQUIRE_MESSAGE(env["ok"] == true, name << ": " << env.dump());
    return env["result"];
}

std::string unb64(const std::string& in) {
    std::string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        int d;
        if (c >= 'A' && c <= 'Z') d = c - 'A';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
        else if (c >= '0' && c <= '9') d = c - '0' + 52;
        else if (c == '+') d = 62;
        else if (c == '/') d = 63;
        else break;
        val = (val << 6) + d;
        bits += 6;
        if (bits >= 0) {
            out.push_back(char((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
    std::string png;
    size_t colours() const {
        std::set<uint32_t> c;
        for (size_t i = 0; i + 3 < px.size() && c.size() <= 4096; i += 4) {
            uint32_t v;
            std::memcpy(&v, &px[i], 4);
            c.insert(v);
        }
        return c.size();
    }
};

Image decode(const json& result) {
    Image img;
    img.png = unb64(result["png_base64"].get<std::string>());
    int n = 0;
    unsigned char* px =
        stbi_load_from_memory((const unsigned char*)img.png.data(), (int)img.png.size(), &img.w, &img.h, &n, 4);
    REQUIRE(px != nullptr);
    img.px.assign(px, px + (size_t)img.w * img.h * 4);
    stbi_image_free(px);
    return img;
}

// The live entries of the catalog, by id.
std::map<std::string, json> live_entries(uint16_t port, const std::string& token) {
    const json listed = ok(port, token, "catalog.list", {{"kind", "live"}});
    std::map<std::string, json> out;
    for (const json& e : listed["entries"]) out[e["id"].get<std::string>()] = e;
    return out;
}

// ---- the hidden app driven by CALIPER_TEST_SPEC_UI, beside a producer ----

struct Env {
    std::string exe, producer, instruments;
    std::string window, shots;   // CALIPER_TEST_WINDOW, CALIPER_TEST_SHOTS ("" when unset)
    bool ok = false;
};

Env test_env() {
    Env e;
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* producer = std::getenv("CALIPER_TEST_PRODUCER_EXE");
    const char* instruments = std::getenv("CALIPER_TEST_INSTRUMENTS");
    if (!exe || !*exe || !producer || !*producer || !instruments || !*instruments) return e;
    e.exe = exe;
    e.producer = producer;
    e.instruments = instruments;
    if (const char* w = std::getenv("CALIPER_TEST_WINDOW"); w && *w) e.window = w;
    if (const char* d = std::getenv("CALIPER_TEST_SHOTS"); d && *d) e.shots = d;
    e.ok = true;
    return e;
}

// A producer: its hub's stream link (with its token) once it listens.
// `stream_dir`, when not empty, is where it writes its stream file.
std::string start_producer(Proc& p, const Env& env, const std::string& stream_dir = {}) {
    std::vector<std::string> args = {"--layers", "4", "--heads", "4", "--tokens", "32", "--hz", "8"};
    if (!stream_dir.empty()) {
        args.push_back("--stream-file");
        args.push_back(stream_dir);
    }
    REQUIRE(p.start(env.producer, args, {}));
    REQUIRE_MESSAGE(p.read_until("[producer] hub ", 20000), p.text);
    REQUIRE(p.read_until("\n", 2000));
    const size_t at = p.text.find("[producer] hub ") + std::string("[producer] hub ").size();
    return p.text.substr(at, p.text.find('\n', at) - at);
}

// The input script's end: "[ui] done" or "[ui] stopped: ...". False after `ms`.
bool script_ended(Proc& app, int ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (app.text.find("[ui] done") != std::string::npos || app.text.find("[ui] stopped") != std::string::npos)
            return true;
        app.read_some(20);
    }
    return false;
}

// Waits until the app's output holds `needle`, reading as it goes.
bool app_says(Proc& app, const std::string& needle, int ms = 30000) { return app.read_until(needle, ms); }

// The token of the app's start line ("with the session token <t>").
std::string start_token(const Proc& app) {
    const std::string mark = "with the session token ";
    const size_t at = app.text.find(mark);
    if (at == std::string::npos) return {};
    size_t e = at + mark.size();
    while (e < app.text.size() &&
           (std::isalnum((unsigned char)app.text[e]) || app.text[e] == '_' || app.text[e] == '-'))
        ++e;
    return app.text.substr(at + mark.size(), e - at - mark.size());
}

// A shot's name for CALIPER_TEST_SHOTS: fix-<what>-<W>x<H>.
std::string shot_name(const Env& env, const std::string& what) {
    return "fix-" + what + "-" + (env.window.empty() ? std::string("800x500") : env.window);
}

// The app's environment: data, workspace's instruments and window size, the
// input script and its shots folder.
std::vector<std::string> app_env(const Env& env, const TempDir& t, const std::string& script,
                                 const std::string& shots) {
    return {"CALIPER_DATA_DIR=" + t.str("data"), "CALIPER_USER_INSTRUMENTS_DIR=" + t.str("user-instruments"),
            "CALIPER_WINDOW_SIZE=" + (env.window.empty() ? std::string("800x500") : env.window),
            "CALIPER_TEST_SPEC_UI=" + script, "CALIPER_TEST_SPEC_UI_SHOTS=" + shots};
}

// Skips the case without Metal; true when it runs.
bool has_metal(Proc& app) {
    app.read_until("[renderer] ", 30000);
    app.read_until("\n", 2000);
    if (app.text.find("[renderer] metal") != std::string::npos) return true;
    MESSAGE("caliper --hidden: run skipped: no GPU (no Metal device)");
    return false;
}

// Waits until `pred` holds, reading the app's output meanwhile.
template <typename F>
bool eventually(Proc& app, F pred, int ms = 20000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (pred()) return true;
        app.read_some(50);
    }
    return pred();
}

}  // namespace

TEST_CASE("app live: F4 lists a caliper.serve() stream from its stream file, and Enter attaches it without its link") {
    const Env env = test_env();
    if (!env.ok) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE, CALIPER_TEST_PRODUCER_EXE or CALIPER_TEST_INSTRUMENTS is not set: run the "
                "ctest entry caliper_app_live_f4_stream");
        return;
    }
    TempDir t("f4");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    Proc producer;
    const std::string link = start_producer(producer, env, t.str("data"));
    const std::string token = link.substr(link.find("token=") + 6);
    REQUIRE(token.size() >= 16);
    const std::string list_shot = shot_name(env, "f4-stream");
    const std::string live_shot = shot_name(env, "f4-stream-attached");

    // The person's steps: F4 on the landing, Enter on the listed stream.
    Proc app;
    REQUIRE(app.start(env.exe, {"--hidden", "--instruments", env.instruments, "--workspace", t.str("ws")},
                      app_env(env, t,
                              "page landing;key f4;wait 10;shot " + list_shot + ";key enter;page live;wait 40;shot " +
                                  live_shot + ";quit",
                              shots)));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(script_ended(app, 120000), app.text);
    CHECK_MESSAGE(app.text.find("[ui] done") != std::string::npos, app.text);
    CHECK(app.text.find("[ui] page live") != std::string::npos);
    CHECK(app.text.find("[live] attached: attention_train.py") != std::string::npos);
    CHECK(fs::exists(fs::path(shots) / (list_shot + ".png")));
    CHECK(fs::exists(fs::path(shots) / (live_shot + ".png")));
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    // The listing gave no more than the link: the token never shows in what the
    // app writes (its Activity and stderr name the stream without it).
    CHECK(app.text.find(token) == std::string::npos);
    // The producer removes its stream file when it stops.
    const std::string stream_file = t.str("data/streams/" + std::to_string(producer.pid) + ".json");
    CHECK(fs::exists(stream_file));
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
    CHECK_FALSE(fs::exists(stream_file));
}

TEST_CASE("app live: the stream stays attached across Home and the spec page until Detach, and a spec over live "
          "sources keeps its data after Home and back") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("home");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    Proc producer;
    const std::string link = start_producer(producer, env);
    const std::string back_shot = shot_name(env, "home-attached");
    const std::string live_shot = shot_name(env, "live-detach");
    const std::string script =
        "page live;key escape;page landing;wait 10;shot " + back_shot + ";touch " + t.str("home") +
        ";wait-file " + t.str("spec-ready") + ";page spec;key escape;page landing;touch " + t.str("home2") +
        ";wait-file " + t.str("back") + ";key escape;page landing;key f5;page live;wait 20;shot " + live_shot + ";touch " +
        t.str("live2") + ";wait-file " + t.str("detach") + ";key cmd+d;page landing;touch " + t.str("detached");
    Proc app;
    REQUIRE(app.start(env.exe,
                      {"--hidden", "--instruments", env.instruments, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      app_env(env, t, script, shots)));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE(token.size() >= 22);

    // Home from the live page: the stream stays attached, and its sources stay
    // in the catalog and keep arriving on the landing screen.
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("home")); }, 60000), app.text);
    std::string updated;
    REQUIRE(eventually(app, [&] {
        const auto live = live_entries(port, token);
        if (!live.count("live:attn.entropy")) return false;
        updated = live.at("live:attn.entropy").value("updated_at", std::string());
        return !updated.empty();
    }));
    CHECK(eventually(app, [&] {
        const auto live = live_entries(port, token);
        return live.count("live:attn.entropy") &&
               live.at("live:attn.entropy").value("updated_at", std::string()) != updated;
    }));
    CHECK(app.text.find("[live] detached") == std::string::npos);

    // A spec over a live source, opened on the spec page, then Home and back:
    // its panel follows the stream both times.
    ok(port, token, "spec.new", {{"id", "entropy"}, {"title", "Entropy"}});
    ok(port, token, "source.add", {{"id", "entropy"}, {"catalog", "live:attn.entropy"}});
    ok(port, token, "panel.create",
       {{"id", "heads"}, {"primitive", "heatmap"}, {"title", "Entropy by layer and head"}, {"bind", {{"values", "entropy"}}}});
    ok(port, token, "instrument.save");
    CHECK(ok(port, token, "instrument.open", {{"id", "entropy"}})["title"] == "Entropy");
    auto renders_change = [&] {
        const Image a = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 320}, {"height", 200}}));
        return eventually(app, [&] {
            const Image b = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 320}, {"height", 200}}));
            return b.colours() > 16 && b.px != a.px;
        });
    };
    CHECK(renders_change());
    std::ofstream(t.str("spec-ready")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("home2")); }, 60000), app.text);
    CHECK(app.text.find("[live] detached") == std::string::npos);
    CHECK(live_entries(port, token).count("live:attn.entropy") == 1);
    CHECK(ok(port, token, "instrument.open", {{"id", "entropy"}})["title"] == "Entropy");
    CHECK(renders_change());

    // F5 on the landing returns to the live page of the attached stream; ⌘D
    // Detach there ends it, and the spec's live panel goes offline.
    const Image last_live = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 320}, {"height", 200}}));
    std::ofstream(t.str("back")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("live2")); }, 60000), app.text);
    CHECK(app.text.find("[live] detached") == std::string::npos);
    std::ofstream(t.str("detach")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("detached")); }, 60000), app.text);
    CHECK(app_says(app, "[live] detached: attention_train.py", 10000));
    CHECK(eventually(app, [&] { return live_entries(port, token).empty(); }));
    CHECK(ok(port, token, "instrument.open", {{"id", "entropy"}})["title"] == "Entropy");
    const Image offline = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 320}, {"height", 200}}));
    CHECK(offline.px != last_live.px);
    REQUIRE(script_ended(app, 30000));
    CHECK_MESSAGE(app.text.find("[ui] done") != std::string::npos, app.text);
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
}

TEST_CASE("app live: a second stream replaces the first; a stream that ends while the person is elsewhere keeps its "
          "last data; quitting while attached detaches") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("second");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    Proc first;
    const std::string link = start_producer(first, env);
    const std::string ended_shot = shot_name(env, "home-ended");
    const std::string script = "page live;key escape;page landing;wait-file " + t.str("second-ready") +
                               ";key f4;wait 10;key enter;page live;wait 10;key escape;page landing;touch " +
                               t.str("home") + ";wait-file " + t.str("ended") + ";wait 20;shot " + ended_shot +
                               ";touch " + t.str("shot");
    Proc app;
    REQUIRE(app.start(env.exe,
                      {"--hidden", "--instruments", env.instruments, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      app_env(env, t, script, shots)));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE(token.size() >= 22);
    REQUIRE(app_says(app, "[live] attached: attention_train.py (pid " + std::to_string(first.pid) + ")"));

    // A second stream, listed by its stream file: Enter on it in F4 detaches
    // the first and attaches the second.
    Proc second;
    start_producer(second, env, t.str("data"));
    const pid_t second_pid = second.pid;   // Proc::wait() forgets it
    std::ofstream(t.str("second-ready")).put('\n');
    REQUIRE_MESSAGE(app_says(app, "[live] attached: attention_train.py (pid " + std::to_string(second_pid) + ")", 60000),
                    app.text);
    CHECK(app.text.find("[live] detached: attention_train.py (pid " + std::to_string(first.pid) + ")") !=
          std::string::npos);
    const std::string session = "attention-" + std::to_string(second_pid);
    CHECK(eventually(app, [&] {
        const auto live = live_entries(port, token);
        return live.count("live:attn") && live.at("live:attn")["origin"].value("session", std::string()) == session;
    }));

    // The second stream ends while the landing shows: the session ended, its
    // sources keep their last data, and nothing detaches.
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("home")); }, 60000), app.text);
    ::kill(second.pid, SIGTERM);
    CHECK(second.wait(10000) == 0);
    REQUIRE_MESSAGE(app_says(app, "[live] session ended: attention_train.py"), app.text);
    CHECK(live_entries(port, token).count("live:attn") == 1);
    std::ofstream(t.str("ended")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("shot")); }, 60000), app.text);
    const size_t ended_at = app.text.find("[live] session ended");
    CHECK(app.text.find("[live] detached: attention_train.py (pid " + std::to_string(second_pid) + ")") ==
          std::string::npos);

    // Quitting while attached detaches on the way out and exits cleanly.
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    CHECK_MESSAGE(app.text.find("[live] detached: attention_train.py (pid " + std::to_string(second_pid) + ")",
                                ended_at) != std::string::npos,
                  app.text.substr(ended_at == std::string::npos ? 0 : ended_at));
    ::kill(first.pid, SIGTERM);
    CHECK(first.wait(10000) == 0);
}

TEST_CASE("app live: a second attach that fails leaves the attached stream attached") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("restore");
    fs::create_directories(t.str("data/streams"));
    fs::permissions(t.str("data/streams"), fs::perms::owner_all, fs::perm_options::replace);
    fs::create_directories(t.str("user-instruments"));
    Proc producer;
    const std::string link = start_producer(producer, env);
    const std::string script = "page live;key escape;page landing;wait-file " + t.str("r0") +
                               ";key f4;wait 10;key enter;wait 30;touch " + t.str("r1");
    Proc app;
    REQUIRE(app.start(env.exe,
                      {"--hidden", "--instruments", env.instruments, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      app_env(env, t, script, t.str(""))));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const uint16_t port = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false)["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE(eventually(app, [&] { return live_entries(port, token).count("live:attn") == 1; }));
    // A stale row: a stream file of a running process (this test) whose port
    // nobody listens on any more.
    uint16_t dead_port = 0;
    {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
        socklen_t len = sizeof addr;
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
        dead_port = ntohs(addr.sin_port);
        ::close(fd);
    }
    char host[256] = {0};
    ::gethostname(host, sizeof host - 1);
    const long now_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const std::string stale = t.str("data/streams/" + std::to_string(::getpid()) + ".json");
    {
        std::ofstream f(stale);
        f << json{{"version", 1}, {"pid", ::getpid()}, {"host", host}, {"session", "stale-1"}, {"script", "stale.py"},
                  {"start_unix_ms", now_ms}, {"port", dead_port},
                  {"stream_url", "ws://127.0.0.1:" + std::to_string(dead_port) + "/v3/stream?token=abcdefghijklmnopqrstuvwxyz0123"},
                  {"control", false}, {"views", json::array()}}
                 .dump();
    }
    fs::permissions(stale, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
    std::ofstream(t.str("r0")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("r1")); }, 60000), app.text);
    CHECK(app_says(app, "attach failed", 10000));
    // The first stream is still attached, and its sources still arrive.
    CHECK(eventually(app, [&] { return live_entries(port, token).count("live:attn") == 1; }));
    fs::remove(stale);
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
}

TEST_CASE("app live: CALIPER_HOME_AFTER goes Home once, and the live page opened again with F5 stays") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("home-after");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    Proc producer;
    const std::string link = start_producer(producer, env);
    // The hook goes Home a second after the live page opened; F5 comes back,
    // and the page stays for the 120 frames after.
    std::vector<std::string> e = app_env(env, t, "page live;page landing;key f5;page live;wait 120;page live;touch " +
                                                     t.str("stayed"), t.str(""));
    e.push_back("CALIPER_HOME_AFTER=1");
    Proc app;
    REQUIRE(app.start(env.exe, {"--hidden", "--instruments", env.instruments, "--url", link, "--workspace", t.str("ws")}, e));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("stayed")); }, 90000), app.text);
    CHECK(app.text.find("[ui] stopped") == std::string::npos);
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
}

TEST_CASE("app live: files dropped on the live page start a new spec beside the live sources, or go into an open "
          "spec with unsaved changes") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("drop");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    for (const char* name : {"runs_a", "runs_b", "runs_c"}) {
        std::ofstream f(t.str(std::string(name) + ".csv"));
        f << "step,loss\n";
        for (int i = 0; i < 20; ++i) f << i << "," << 1.0 / (i + 1) << "\n";
    }
    Proc producer;
    const std::string link = start_producer(producer, env);
    const std::string drop_shot = shot_name(env, "live-drop");
    // A drop on the live page; then, with the new spec unsaved, a second drop
    // from the live page goes into it; saved, a third drop starts another.
    const std::string script =
        "page live;drop " + t.str("runs_a.csv") + ";page spec;pane source table:runs_a;pane source live:attn.entropy;" +
        "wait 20;shot " + drop_shot + ";touch " + t.str("first") + ";key escape;wait 5;key escape;page landing;key f5;page live;drop " +
        t.str("runs_b.csv") + ";page spec;pane source table:runs_b;touch " + t.str("second") + ";wait-file " +
        t.str("saved") + ";key escape;wait 5;key escape;page landing;key f5;page live;drop " + t.str("runs_c.csv") +
        ";page spec;pane source table:runs_c;touch " + t.str("third");
    Proc app;
    REQUIRE(app.start(env.exe,
                      {"--hidden", "--instruments", env.instruments, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      app_env(env, t, script, shots)));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE(token.size() >= 22);

    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("first")); }, 90000), app.text);
    CHECK(app.text.find("drop refused") == std::string::npos);
    const json first = ok(port, token, "spec.get");
    const std::string first_id = first["id"].get<std::string>();
    CHECK(first["dirty"] == true);
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("second")); }, 90000), app.text);
    CHECK(ok(port, token, "spec.get")["id"] == first_id);   // the same, unsaved spec
    ok(port, token, "instrument.save");
    std::ofstream(t.str("saved")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("third")); }, 90000), app.text);
    CHECK(ok(port, token, "spec.get")["id"] != first_id);   // a new spec this time
    REQUIRE(script_ended(app, 30000));
    CHECK_MESSAGE(app.text.find("[ui] done") != std::string::npos, app.text);
    CHECK(app.text.find("[live] detached") == std::string::npos);
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
}

// ---- "Save changes to <name>?" (src/host/builder/save_prompt.h) ----

namespace {

// The hidden app on an empty workspace with Caliper API, and its API port and
// the person's token; false when it runs without Metal (skipped).
struct SaveApp {
    Proc app;
    uint16_t port = 0;
    std::string token;
    bool start(const Env& env, const TempDir& t, const std::string& script, const std::string& shots) {
        REQUIRE(app.start(env.exe, {"--hidden", "--instruments", env.instruments, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                          app_env(env, t, script, shots)));
        if (!has_metal(app)) {
            ::kill(app.pid, SIGTERM);
            app.wait(30000);
            return false;
        }
        REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
        const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
        REQUIRE(api.is_object());
        port = api["port"].get<uint16_t>();
        token = start_token(app);
        REQUIRE(token.size() >= 22);
        return true;
    }
    json spec() { return ok(port, token, "spec.get"); }
    void reach(const TempDir& t, const std::string& file) {
        REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str(file)); }, 90000), app.text);
    }
    size_t count(const std::string& needle) const {
        size_t n = 0;
        for (size_t at = app.text.find(needle); at != std::string::npos; at = app.text.find(needle, at + 1)) ++n;
        return n;
    }
};

void go_on(const TempDir& t, const std::string& file) { std::ofstream(t.str(file)).put('\n'); }

// The open spec's text with its title changed, as an edit that leaves it unsaved.
void retitle(SaveApp& a, const std::string& from, const std::string& to) {
    std::string text = a.spec()["text"].get<std::string>();
    const std::string old_line = "title = \"" + from + "\"";
    const size_t at = text.find(old_line);
    REQUIRE_MESSAGE(at != std::string::npos, text);
    text.replace(at, old_line.size(), "title = \"" + to + "\"");
    ok(a.port, a.token, "spec.set_text", {{"text", text}});
}

}  // namespace

TEST_CASE("app save: F3 New dashboard over unsaved changes asks Save changes first; Cancel keeps the spec, Don't "
          "Save and Save go on") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("save-new");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    const std::string shot = shot_name(env, "save-changes");
    const std::string home = "key escape;wait 5;key escape;page landing;";
    auto new_spec = [](const std::string& title) {
        return "key cmd+k;wait 5;type new " + title + ";wait 5;key enter;";
    };
    const std::string script = "page landing;wait 60;" + new_spec("Alpha") + "page spec;touch " + t.str("a1") + ";" + home +
                               new_spec("Beta") + "save until;touch " + t.str("a2") + ";shot " + shot +
                               ";save cancel;wait 10;touch " + t.str("a3") + ";" + new_spec("Beta") +
                               "save dont-save;page spec;touch " + t.str("a4") + ";" + home + new_spec("Gamma") +
                               "save save;page spec;touch " + t.str("a5");
    SaveApp a;
    if (!a.start(env, t, script, shots)) return;

    a.reach(t, "a1");
    CHECK(a.spec()["id"] == "alpha");
    CHECK(a.spec()["dirty"] == true);
    // F3 again: the question holds the new spec back.
    a.reach(t, "a2");
    CHECK(a.spec()["id"] == "alpha");
    CHECK(app_says(a.app, "[save] Save changes to Alpha? waiting", 10000));
    // Cancel: nothing changed.
    a.reach(t, "a3");
    CHECK(a.spec()["id"] == "alpha");
    CHECK(a.spec()["dirty"] == true);
    // Don't Save: the new spec, and Alpha was never written.
    a.reach(t, "a4");
    CHECK(a.spec()["id"] == "beta");
    CHECK_FALSE(fs::exists(t.str("ws/instruments/alpha")));
    // Save: Beta is saved, then the new spec.
    a.reach(t, "a5");
    CHECK(a.spec()["id"] == "gamma");
    CHECK(fs::exists(t.str("ws/instruments/beta/beta.spec.toml")));
    CHECK(fs::exists(fs::path(shots) / (shot + ".png")));
    REQUIRE(script_ended(a.app, 30000));
    CHECK_MESSAGE(a.app.text.find("[ui] done") != std::string::npos, a.app.text);
    // A hidden app's SIGTERM never asks, with Gamma unsaved.
    const size_t asked = a.count("[save] Save changes to");
    ::kill(a.app.pid, SIGTERM);
    CHECK_MESSAGE(a.app.wait(30000) == 0, a.app.text);
    CHECK(a.count("[save] Save changes to") == asked);
}

TEST_CASE("app save: opening another dashboard from the landing over unsaved changes asks first; the open spec's "
          "own dashboard shows its changes; an agent's launch never asks") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("save-open");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    const std::string home = "key escape;wait 5;key escape;page landing;";
    const std::string open_two = "key cmd+k;wait 5;type Two;wait 5;key enter;";
    const std::string script = "wait-file " + t.str("b0") + ";page spec;" + home + open_two + "save until;touch " +
                               t.str("b1") + ";save save;page spec;wait 10;touch " + t.str("b2") + ";wait-file " +
                               t.str("b2go") + ";" + home + open_two + "page spec;wait 30;touch " + t.str("b3");
    SaveApp a;
    if (!a.start(env, t, script, shots)) return;
    ok(a.port, a.token, "spec.new", {{"id", "one"}, {"title", "One"}});
    ok(a.port, a.token, "instrument.save");
    ok(a.port, a.token, "spec.new", {{"id", "two"}, {"title", "Two"}});
    ok(a.port, a.token, "instrument.save");
    CHECK(ok(a.port, a.token, "instrument.open", {{"id", "one"}})["title"] == "One");
    retitle(a, "One", "One edited");
    CHECK(a.spec()["dirty"] == true);
    go_on(t, "b0");

    // Two from the landing: the question first, and One stays open meanwhile.
    a.reach(t, "b1");
    CHECK(a.spec()["id"] == "one");
    CHECK(app_says(a.app, "[save] Save changes to One edited? waiting", 10000));
    // Save: One is written with its change, then Two opens.
    a.reach(t, "b2");
    CHECK(a.spec()["id"] == "two");
    {
        std::ifstream f(t.str("ws/instruments/one/one.spec.toml"));
        const std::string saved((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        CHECK(saved.find("One edited") != std::string::npos);
    }
    // Two edited, then Two again from the landing: its own page, changes and
    // all, and no question.
    retitle(a, "Two", "Two edited");
    const size_t asked = a.count("[save] Save changes to");
    go_on(t, "b2go");
    a.reach(t, "b3");
    CHECK(a.spec()["id"] == "two");
    CHECK(a.spec()["dirty"] == true);
    CHECK(a.spec()["text"].get<std::string>().find("Two edited") != std::string::npos);
    CHECK(a.count("[save] Save changes to") == asked);
    // An agent's launch (Caliper API here) is not the person's: no question.
    CHECK(ok(a.port, a.token, "app.launch", {{"id", "local.one"}})["id"] == "local.one");   // the manifest's id
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "one"; }));
    CHECK(a.count("[save] Save changes to") == asked);
    // The person's own launches asked the question and put up no card.
    CHECK(a.app.text.find("[approval]") == std::string::npos);
    REQUIRE(script_ended(a.app, 30000));
    CHECK_MESSAGE(a.app.text.find("[ui] done") != std::string::npos, a.app.text);
    ::kill(a.app.pid, SIGTERM);
    CHECK_MESSAGE(a.app.wait(30000) == 0, a.app.text);
}

TEST_CASE("app save: closing the window over unsaved changes asks first; Cancel keeps the app running, Don't Save "
          "quits") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("save-close");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    const std::string script = "wait-file " + t.str("c0") + ";close;save until;touch " + t.str("c1") +
                               ";save cancel;wait 30;touch " + t.str("c2") + ";wait-file " + t.str("c2go") +
                               ";close;save dont-save";
    SaveApp a;
    if (!a.start(env, t, script, shots)) return;
    ok(a.port, a.token, "spec.new", {{"id", "draft"}, {"title", "Draft"}});
    go_on(t, "c0");
    a.reach(t, "c1");
    CHECK(::kill(a.app.pid, 0) == 0);   // still running: the question holds the close back
    CHECK(app_says(a.app, "[save] Save changes to Draft? waiting", 10000));
    a.reach(t, "c2");
    CHECK(::kill(a.app.pid, 0) == 0);   // Cancel: it runs on
    CHECK(a.spec()["id"] == "draft");
    go_on(t, "c2go");
    CHECK_MESSAGE(a.app.wait(60000) == 0, a.app.text);   // Don't Save: it quits
    CHECK(a.app.text.find("[save] Don't Save") != std::string::npos);
    CHECK_FALSE(fs::exists(t.str("ws/instruments/draft")));
}

// ---- an agent's launch of a dashboard over unsaved changes ----

namespace {

// POST /v0/instruments/<id>/launch with `token`.
Http launch_v0(uint16_t port, const std::string& token, const std::string& id) {
    return http(port, "POST", "/v0/instruments/" + id + "/launch", token);
}

// The app lists the workspace's instrument `id` (after an instrument.save it rescans).
bool lists(SaveApp& a, const std::string& id) {
    return eventually(a.app, [&] {
        const Http r = http(a.port, "GET", "/v0/instruments", a.token);
        return r.status == 200 && r.body.find("\"" + id + "\"") != std::string::npos;
    });
}

}  // namespace

TEST_CASE("app save: closing the window while a command runs on the session's worker still asks Save changes once "
          "it ends") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("save-busy");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string script = "wait-file " + t.str("c0") + ";worker busy;close;wait 20;touch " + t.str("c1") +
                               ";worker idle;save until;touch " + t.str("c2") + ";save cancel;wait 10;touch " +
                               t.str("c3");
    SaveApp a;
    if (!a.start(env, t, script, t.str(""))) return;
    ok(a.port, a.token, "spec.new", {{"id", "draft"}, {"title", "Draft"}});
    // A query that keeps the worker busy until its own time limit, as an
    // agent's data.query, data.import or loader.run does.
    auto query = std::async(std::launch::async, [&] {
        return http(a.port, "POST", "/v0/commands/data.query", a.token,
                    json{{"sql", "SELECT count(*) FROM range(100000000000) t(i) WHERE i % 7 = 3"},
                         {"timeout_ms", 6000}}.dump());
    });
    go_on(t, "c0");
    // Closed while the worker runs: the app does not quit; it waits for the
    // command, then asks about the unsaved spec.
    a.reach(t, "c1");
    CHECK(::kill(a.app.pid, 0) == 0);
    CHECK(app_says(a.app, "[save] Save changes to Draft? waiting", 30000));
    a.reach(t, "c2");
    a.reach(t, "c3");
    CHECK(::kill(a.app.pid, 0) == 0);   // Cancel: it runs on
    CHECK(a.spec()["id"] == "draft");
    CHECK(query.get().status == 200);
    ::kill(a.app.pid, SIGTERM);
    CHECK_MESSAGE(a.app.wait(30000) == 0, a.app.text);
}

TEST_CASE("app launch: an outside agent's launch of a dashboard over unsaved changes waits on an approval card; "
          "over a saved spec, and with the person's own token, it opens at once") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("launch-mcp");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string decisions = t.str("decisions");
    std::ofstream(decisions).flush();
    SaveApp a;
    {
        std::vector<std::string> e = app_env(env, t, "wait 1", t.str(""));
        e.push_back("CALIPER_TEST_APPROVALS=" + decisions);
        REQUIRE(a.app.start(env.exe,
                            {"--hidden", "--instruments", env.instruments, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                            e));
        if (!has_metal(a.app)) {
            ::kill(a.app.pid, SIGTERM);
            a.app.wait(30000);
            return;
        }
        REQUIRE_MESSAGE(a.app.read_until("[api] Instruments in a browser", 60000), a.app.text);
        const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
        REQUIRE(api.is_object());
        a.port = api["port"].get<uint16_t>();
        a.token = start_token(a.app);
        REQUIRE(a.token.size() >= 22);
    }
    // The API file's token is an outside agent's, origin mcp.
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    const std::string mcp = api["token"].get<std::string>();
    REQUIRE(mcp != a.token);
    ok(a.port, a.token, "spec.new", {{"id", "one"}, {"title", "One"}});
    ok(a.port, a.token, "instrument.save");
    ok(a.port, a.token, "spec.new", {{"id", "two"}, {"title", "Two"}});
    ok(a.port, a.token, "instrument.save");
    REQUIRE(lists(a, "local.one"));
    REQUIRE(lists(a, "local.two"));
    CHECK(ok(a.port, a.token, "instrument.open", {{"id", "one"}})["title"] == "One");
    CHECK(a.spec()["dirty"] == false);
    auto cards = [&](const std::string& origin) { return a.count("instrument.open (" + origin + ") waiting"); };

    // Over a saved spec: an agent's launch opens at once, with no card.
    REQUIRE(launch_v0(a.port, mcp, "local.two").status == 200);
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "two"; }));
    CHECK(cards("mcp") == 0);

    // Over unsaved changes: held on a card, and nothing changes until it is
    // approved.
    retitle(a, "Two", "Two edited");
    REQUIRE(launch_v0(a.port, mcp, "local.one").status == 200);
    REQUIRE_MESSAGE(eventually(a.app, [&] { return cards("mcp") == 1; }), a.app.text);
    CHECK(a.spec()["id"] == "two");
    CHECK(a.spec()["dirty"] == true);
    std::ofstream(decisions, std::ios::app) << "approve instrument.open mcp\n";
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "one"; }));

    // The same through the command layer's app.launch (caliper-mcp's way),
    // denied this time: the open spec stays, unsaved.
    retitle(a, "One", "One edited");
    {
        const Http r = http(a.port, "POST", "/v0/commands/app.launch", mcp, json{{"id", "local.two"}}.dump());
        REQUIRE(r.status == 200);
        CHECK_MESSAGE(json::parse(r.body)["ok"] == true, r.body);
    }
    REQUIRE_MESSAGE(eventually(a.app, [&] { return cards("mcp") == 2; }), a.app.text);
    std::ofstream(decisions, std::ios::app) << "deny instrument.open mcp\n";
    CHECK(eventually(a.app, [&] { return a.app.text.find("instrument.open (mcp) denied") != std::string::npos; }));
    CHECK(a.spec()["id"] == "one");
    CHECK(a.spec()["dirty"] == true);

    // The person's own token (origin api, not under review): at once, as
    // the design says, and ⌘Z would bring the unsaved spec back.
    REQUIRE(launch_v0(a.port, a.token, "local.two").status == 200);
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "two"; }));
    CHECK(cards("api") == 0);
    ::kill(a.app.pid, SIGTERM);
    CHECK_MESSAGE(a.app.wait(30000) == 0, a.app.text);
}

namespace {

// The approval id of the newest card line "[approval] <id> <name> (<origin>) waiting".
std::string waiting_card(const Proc& app, const std::string& name, const std::string& origin) {
    const std::string tail = " " + name + " (" + origin + ") waiting";
    const size_t at = app.text.rfind(tail);
    if (at == std::string::npos) return {};
    const size_t start = app.text.rfind("[approval] ", at);
    if (start == std::string::npos) return {};
    return app.text.substr(start + 11, at - start - 11);
}

}  // namespace

TEST_CASE("app launch: the answer to an outside agent's launch over unsaved work says it waits for the person's "
          "approval, and the approval can be looked up") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("launch-reply");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string decisions = t.str("decisions");
    std::ofstream(decisions).flush();
    SaveApp a;
    {
        std::vector<std::string> e = app_env(env, t, "wait 1", t.str(""));
        e.push_back("CALIPER_TEST_APPROVALS=" + decisions);
        REQUIRE(a.app.start(env.exe,
                            {"--hidden", "--instruments", env.instruments, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                            e));
        if (!has_metal(a.app)) {
            ::kill(a.app.pid, SIGTERM);
            a.app.wait(30000);
            return;
        }
        REQUIRE_MESSAGE(a.app.read_until("[api] Instruments in a browser", 60000), a.app.text);
        a.port = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false)["port"].get<uint16_t>();
        a.token = start_token(a.app);
        REQUIRE(a.token.size() >= 22);
    }
    const std::string mcp = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false)["token"].get<std::string>();
    ok(a.port, a.token, "spec.new", {{"id", "one"}, {"title", "One"}});
    ok(a.port, a.token, "instrument.save");
    ok(a.port, a.token, "spec.new", {{"id", "two"}, {"title", "Two"}});
    ok(a.port, a.token, "instrument.save");
    REQUIRE(lists(a, "local.one"));
    REQUIRE(lists(a, "local.two"));
    CHECK(ok(a.port, a.token, "instrument.open", {{"id", "one"}})["title"] == "One");

    // Not held (a saved spec): the answer is as before.
    {
        const Http r = launch_v0(a.port, mcp, "local.two");
        REQUIRE(r.status == 200);
        const json j = json::parse(r.body);
        CHECK(j["status"] == "ACTIVE");
        CHECK_FALSE(j.contains("note"));
    }
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "two"; }));

    // Held, through POST /v0/instruments/{id}/launch: PENDING_APPROVAL with a note,
    // and the card's approval is pending; approved, the spec is open.
    retitle(a, "Two", "Two edited");
    {
        const Http r = launch_v0(a.port, mcp, "local.one");
        REQUIRE(r.status == 200);
        const json j = json::parse(r.body);
        CHECK_MESSAGE(j["status"] == "PENDING_APPROVAL", r.body);
        CHECK(j["already_open"] == false);
        CHECK(j.value("note", std::string()).find("approval") != std::string::npos);
        CHECK(j.value("note", std::string()).find("spec.get") != std::string::npos);
    }
    std::string id;
    REQUIRE_MESSAGE(eventually(a.app, [&] { return !(id = waiting_card(a.app, "instrument.open", "mcp")).empty(); }),
                    a.app.text);
    {
        const Http r = http(a.port, "GET", "/v0/approvals/" + id, a.token);
        REQUIRE(r.status == 200);
        const json j = json::parse(r.body);
        CHECK(j["state"] == "pending");
        CHECK(j["name"] == "instrument.open");
        CHECK(j["origin"] == "mcp");
    }
    CHECK(a.spec()["id"] == "two");
    std::ofstream(decisions, std::ios::app) << "approve instrument.open mcp\n";
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "one"; }));
    CHECK(eventually(a.app, [&] {
        return json::parse(http(a.port, "GET", "/v0/approvals/" + id, a.token).body)["state"] == "decided";
    }));

    // Held, through app.launch: the same answer; denied, the spec is unchanged.
    retitle(a, "One", "One edited");
    {
        const Http r = http(a.port, "POST", "/v0/commands/app.launch", mcp, json{{"id", "local.two"}}.dump());
        REQUIRE(r.status == 200);
        const json env_ = json::parse(r.body);
        REQUIRE_MESSAGE(env_["ok"] == true, r.body);
        CHECK(env_["result"]["status"] == "PENDING_APPROVAL");
        CHECK(env_["result"].value("note", std::string()).find("approval") != std::string::npos);
    }
    std::string id2;
    REQUIRE_MESSAGE(eventually(a.app, [&] {
                        id2 = waiting_card(a.app, "instrument.open", "mcp");
                        return !id2.empty() && id2 != id;
                    }),
                    a.app.text);
    std::ofstream(decisions, std::ios::app) << "deny instrument.open mcp\n";
    CHECK(eventually(a.app, [&] {
        const json j = json::parse(http(a.port, "GET", "/v0/approvals/" + id2, a.token).body);
        return j["state"] == "decided" && j["result"]["ok"] == false;
    }));
    CHECK(a.spec()["id"] == "one");
    CHECK(a.spec()["dirty"] == true);

    // The person's own token is not reviewed: as before.
    {
        const Http r = launch_v0(a.port, a.token, "local.two");
        REQUIRE(r.status == 200);
        CHECK(json::parse(r.body)["status"] == "ACTIVE");
    }
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "two"; }));
    ::kill(a.app.pid, SIGTERM);
    CHECK_MESSAGE(a.app.wait(30000) == 0, a.app.text);
}

TEST_CASE("app launch: an in-app agent's launch of a dashboard over unsaved changes waits on an approval card") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("launch-agent");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    // A first run saves the dashboard One into the workspace.
    {
        SaveApp first;
        if (!first.start(env, t, "wait 1", t.str(""))) return;
        ok(first.port, first.token, "spec.new", {{"id", "one"}, {"title", "One"}});
        ok(first.port, first.token, "instrument.save");
        ::kill(first.app.pid, SIGTERM);
        REQUIRE(first.app.wait(30000) == 0);
    }
    // The second starts a new spec, unsaved, then launches One from the
    // origin agent, as the side chat's agent's app.launch arrives.
    const std::string script = t.str("script.jsonl");
    std::ofstream(script) << R"({"name": "spec.new", "args": {"id": "draft", "title": "Draft"}})" << "\n"
                          << R"({"name": "app.launch", "args": {"id": "local.one"}, "origin": "agent"})" << "\n";
    const std::string decisions = t.str("decisions");
    std::ofstream(decisions).flush();
    Proc app;
    std::vector<std::string> e = app_env(env, t, "wait 1", t.str(""));
    e.push_back("CALIPER_COMMAND_SCRIPT=" + script);
    e.push_back("CALIPER_TEST_APPROVALS=" + decisions);
    REQUIRE(app.start(env.exe, {"--hidden", "--instruments", env.instruments, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                      e));
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE_MESSAGE(app_says(app, "instrument.open (agent) waiting", 60000), app.text);
    CHECK(ok(port, token, "spec.get")["id"] == "draft");
    std::ofstream(decisions, std::ios::app) << "approve instrument.open agent\n";
    CHECK(eventually(app, [&] { return ok(port, token, "spec.get")["id"] == "one"; }));
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
}

TEST_CASE("app live: attention entropy per layer, from a live stream, on the spec page while it trains") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* producer_exe = std::getenv("CALIPER_TEST_PRODUCER_EXE");
    const char* instruments = std::getenv("CALIPER_TEST_INSTRUMENTS");
    if (!exe || !*exe || !producer_exe || !*producer_exe || !instruments || !*instruments) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE, CALIPER_TEST_PRODUCER_EXE or CALIPER_TEST_INSTRUMENTS is not set: run the "
                "ctest entry caliper_app_live_entropy");
        return;
    }
    const char* window = std::getenv("CALIPER_TEST_WINDOW");
    const char* shots = std::getenv("CALIPER_TEST_SHOTS");
    TempDir t("app");
    const std::string ws = t.str("ws");
    const std::string data = t.str("data");
    fs::create_directories(t.str("user-instruments"));

    // The training script's stand-in: 6 layers of 8 heads over 64 tokens.
    Proc producer;
    REQUIRE(producer.start(producer_exe, {"--layers", "6", "--heads", "8", "--tokens", "64", "--hz", "8"}, {}));
    REQUIRE_MESSAGE(producer.read_until("[producer] hub ", 20000), producer.text);
    REQUIRE(producer.read_until("\n", 2000));
    const size_t at = producer.text.find("[producer] hub ") + std::string("[producer] hub ").size();
    const std::string link = producer.text.substr(at, producer.text.find('\n', at) - at);

    Proc app;
    REQUIRE(app.start(exe,
                      {"--hidden", "--instruments", instruments, "--url", link, "--api", "--api-port", "0", "--workspace", ws},
                      {"CALIPER_DATA_DIR=" + data, "CALIPER_USER_INSTRUMENTS_DIR=" + t.str("user-instruments"),
                       std::string("CALIPER_WINDOW_SIZE=") + (window && *window ? window : "800x500")}));
    REQUIRE_MESSAGE(app.read_until("[api] Caliper API on ", 60000), app.text);
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 10000), app.text);   // after the start line
    if (app.text.find("[renderer] metal") == std::string::npos) {
        MESSAGE("caliper --hidden: run skipped: no GPU (no Metal device)");
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        ::kill(producer.pid, SIGTERM);
        producer.wait(10000);
        return;
    }
    const json api = json::parse(std::ifstream(data + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    // The test builds the spec as the person's own scripting would, with the
    // token of the app's start line, whose commands are not reviewed; the API
    // file's token is an outside agent's (mcp), whose edits wait for review.
    std::string token;
    {
        const std::string mark = "with the session token ";
        const size_t at = app.text.find(mark);
        REQUIRE(at != std::string::npos);
        size_t e = at + mark.size();
        while (e < app.text.size() && (std::isalnum((unsigned char)app.text[e]) || app.text[e] == '_' || app.text[e] == '-')) ++e;
        token = app.text.substr(at + mark.size(), e - at - mark.size());
    }
    REQUIRE(token.size() >= 22);
    CHECK(token != api["token"].get<std::string>());

    // The live sources reach the catalog with their shapes, and with a rate
    // once a few snapshots have arrived.
    std::map<std::string, json> live;
    const auto until = std::chrono::steady_clock::now() + 30s;
    while (std::chrono::steady_clock::now() < until) {
        live = live_entries(port, token);
        if (live.count("live:attn") && live.count("live:attn.entropy") && live["live:attn"]["rate_hz"].is_number() &&
            live["live:attn"]["rate_hz"].get<double>() > 0.0)
            break;
        std::this_thread::sleep_for(200ms);
    }
    INFO(json(live).dump());
    REQUIRE(live.count("live:attn") == 1);
    REQUIRE(live.count("live:attn.entropy") == 1);
    REQUIRE(live.count("live:entropy.l0") == 1);
    // The head-mean attention, one cell per layer, and the entropy per head.
    CHECK(live["live:attn"]["shape"] == json::array({6, 64, 64}));
    CHECK(live["live:attn.entropy"]["shape"] == json::array({6, 8}));
    // A rate exists once snapshots arrived; its value follows the producer's speed.
    CHECK(live["live:attn"]["rate_hz"].get<double>() > 0.0);
    CHECK(live["live:attn"]["wire"] == true);

    // The spec, through commands: the entropy by layer and head, the entropy
    // of each layer's head-mean attention computed by the spec's transforms,
    // and each layer's mean entropy over the steps.
    ok(port, token, "spec.new", {{"id", "attention-entropy"}, {"title", "Attention entropy"}});
    ok(port, token, "source.add", {{"id", "entropy"}, {"catalog", "live:attn.entropy"}});
    ok(port, token, "source.add", {{"id", "attn"}, {"catalog", "live:attn"}});
    ok(port, token, "source.add",
       {{"id", "layer_entropy"},
        {"from", "attn"},
        {"transform",
         {{{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}}, {{"op", "reduce"}, {"axis", -1}, {"fn", "mean"}}}}});
    json history = json::array();
    for (int l = 0; l < 6; ++l) {
        const std::string id = "layer" + std::to_string(l);
        ok(port, token, "source.add", {{"id", id}, {"catalog", "live:entropy.l" + std::to_string(l)}});
        history.push_back(id);
    }
    ok(port, token, "panel.create",
       {{"id", "heads"}, {"primitive", "heatmap"}, {"title", "Attention entropy by layer and head"},
        {"bind", {{"values", "entropy"}}}, {"options", {{"x_label", "Head"}, {"y_label", "Layer"}}}});
    ok(port, token, "panel.create",
       {{"id", "layers"}, {"primitive", "series"}, {"title", "Entropy of the head-mean attention by layer"},
        {"bind", {{"y", "layer_entropy"}}}, {"options", {{"x_label", "Layer"}}}});
    ok(port, token, "panel.create",
       {{"id", "steps"}, {"primitive", "series"}, {"title", "Mean entropy per layer by step"},
        {"bind", {{"y", history}}}, {"options", {{"x_label", "Step"}}}});
    ok(port, token, "layout.set",
       {{"layout",
         {{"root", "main"},
          {"nodes",
           {{{"id", "main"}, {"split", "vertical"}, {"ratio", 0.6}, {"children", {"top", "steps"}}},
            {{"id", "top"}, {"split", "horizontal"}, {"ratio", 0.5}, {"children", {"heads", "layers"}}}}}}}});
    ok(port, token, "instrument.save");
    CHECK(ok(port, token, "instrument.open", {{"id", "attention-entropy"}})["title"] == "Attention entropy");

    // While it trains: two captures and two renders of the heatmap a moment
    // apart differ, and the source's newest snapshot moves on.
    std::this_thread::sleep_for(1500ms);
    const Image a = decode(ok(port, token, "capture.frame"));
    const Image heads_a = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 480}, {"height", 320}}));
    const std::string updated_a = ok(port, token, "catalog.describe", {{"id", "live:attn.entropy"}})["entry"]
                                      .value("updated_at", std::string());
    std::this_thread::sleep_for(3000ms);
    const Image b = decode(ok(port, token, "capture.frame"));
    const Image heads_b = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 480}, {"height", 320}}));
    const std::string updated_b = ok(port, token, "catalog.describe", {{"id", "live:attn.entropy"}})["entry"]
                                      .value("updated_at", std::string());
    CHECK(a.colours() > 16);
    CHECK(b.colours() > 16);
    CHECK(a.w == b.w);
    CHECK(a.px != b.px);
    CHECK(heads_a.colours() > 16);
    CHECK(heads_a.px != heads_b.px);
    CHECK_FALSE(updated_a.empty());
    CHECK(updated_a != updated_b);
    // The derived source resolves on the live data: one value per layer.
    const json layers_render = ok(port, token, "panel.render", {{"id", "layers"}});
    CHECK(layers_render["width"] == 800);
    if (shots && *shots) {
        std::ofstream(fs::path(shots) / "live-entropy-a.png", std::ios::binary) << a.png;
        std::ofstream(fs::path(shots) / "live-entropy-b.png", std::ios::binary) << b.png;
    }

    // Launching another instrument keeps the stream attached: only Detach on the
    // live page ends it. The catalog keeps the live sources, which keep
    // arriving, and the spec opened again draws live data.
    CHECK(ok(port, token, "app.launch", {{"id", "ai.caliperapp.hello-world"}})["status"] == "ACTIVE");
    CHECK(eventually(app, [&] {
        const auto e = live_entries(port, token);
        return e.count("live:attn.entropy") && e.at("live:attn.entropy").value("updated_at", std::string()) != updated_b;
    }));
    CHECK(app.text.find("[live] detached") == std::string::npos);
    CHECK(ok(port, token, "instrument.open", {{"id", "attention-entropy"}})["title"] == "Attention entropy");
    const Image again = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 480}, {"height", 320}}));
    CHECK(again.colours() > 16);
    const Image page = decode(ok(port, token, "capture.frame"));
    CHECK(page.colours() > 1);

    // Quitting while attached detaches on the way out.
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    CHECK(app.text.find("[live] detached") != std::string::npos);
    CHECK(app.text.find("[session] command instrument.open (api)") != std::string::npos);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
}

// ---- A pan, then an orbit, in a scene panel of the spec page ----------------------------------
// A right-button pan of a scene panel must open no menu, so that the orbit that follows moves the
// camera from its first move: a menu opened by the pan's press took the next press, and that whole
// drag moved nothing. A right click without a drag still opens the panel's menu. The frames come
// from the app's frame log (CALIPER_FRAME_LOG): its mouse buttons, popup, panel and camera columns.

namespace {

struct FrameRow {
    int down = 0, popup = 0, panel_down = 0;
    std::string pointer, panel, camera;
};

std::vector<FrameRow> frame_rows(const std::string& path) {
    std::vector<FrameRow> rows;
    std::ifstream in(path);
    std::string line;
    std::map<std::string, size_t> col;
    auto split = [](const std::string& s) {
        std::vector<std::string> out;
        size_t at = 0;
        for (size_t tab; (tab = s.find('\t', at)) != std::string::npos; at = tab + 1) out.push_back(s.substr(at, tab - at));
        out.push_back(s.substr(at));
        return out;
    };
    if (!std::getline(in, line)) return rows;
    const std::vector<std::string> head = split(line);
    for (size_t i = 0; i < head.size(); ++i) col[head[i]] = i;
    for (const char* name : {"mouse_down", "popup", "panel", "panel_down", "mouse_x", "mouse_y", "camera"})
        if (!col.count(name)) return rows;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = split(line);
        if (f.size() != head.size()) continue;
        FrameRow r;
        r.down = std::atoi(f[col["mouse_down"]].c_str());
        r.popup = std::atoi(f[col["popup"]].c_str());
        r.panel_down = f[col["panel_down"]] == "1" ? 1 : 0;
        r.pointer = f[col["mouse_x"]] + " " + f[col["mouse_y"]];
        r.panel = f[col["panel"]];
        r.camera = f[col["camera"]];
        rows.push_back(std::move(r));
    }
    return rows;
}

// The runs of consecutive frames over the panel with exactly the buttons `bits` held (1 left, 2
// right), as [first, last] row indices.
std::vector<std::pair<size_t, size_t>> button_runs(const std::vector<FrameRow>& rows, const std::string& panel, int bits) {
    std::vector<std::pair<size_t, size_t>> out;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].down != bits || rows[i].panel != panel) continue;
        if (!out.empty() && out.back().second + 1 == i) out.back().second = i;
        else out.push_back({i, i});
    }
    return out;
}

// In a run, the frames whose pointer moved, and those of them whose camera moved too.
std::pair<int, int> moves_in(const std::vector<FrameRow>& rows, std::pair<size_t, size_t> run) {
    int moved = 0, camera = 0;
    for (size_t i = run.first + 1; i <= run.second; ++i) {
        if (rows[i].pointer == rows[i - 1].pointer) continue;
        ++moved;
        if (rows[i].camera != rows[i - 1].camera && rows[i].panel_down == 1) ++camera;
    }
    return {moved, camera};
}

}  // namespace

TEST_CASE("app live: a right-button pan of a scene panel opens no menu, and the orbit after it moves the camera at "
          "every move, with data arriving and after the stream ended; a right click opens the menu") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("pan");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    Proc producer;
    const std::string link = start_producer(producer, env);
    // Each drag moves the pointer once a frame for this many frames (spec_ui_driver.h).
    const int drag_steps = 14;
    const std::string step_words = " steps " + std::to_string(drag_steps) + ";";
    const std::string pan = "drag panel:world 0.4 0.5 to panel:world 0.6 0.55 right" + step_words;
    const std::string orbit = "drag panel:world 0.25 0.5 to panel:world 0.75 0.6" + step_words;
    const std::string script = "wait-file " + t.str("spec-ready") + ";page spec;wait 30;" + pan + orbit + "touch " +
                               t.str("streaming") + ";wait-file " + t.str("ended") + ";wait 30;" + pan + orbit +
                               "menu panel:world;wait 10;touch " + t.str("menu") + ";key escape;wait 5";
    std::vector<std::string> e = app_env(env, t, script, shots);
    e.push_back("CALIPER_FRAME_LOG=" + t.str("frames.tsv"));
    Proc app;
    REQUIRE(app.start(env.exe,
                      {"--hidden", "--instruments", env.instruments, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      e));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        ::kill(producer.pid, SIGTERM);
        producer.wait(10000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE(token.size() >= 22);
    REQUIRE(eventually(app, [&] { return live_entries(port, token).count("live:attn") == 1; }, 60000));

    // A scene panel of its own: the first layer's head-mean attention as a surface.
    ok(port, token, "spec.new", {{"id", "pan"}, {"title", "Pan"}});
    ok(port, token, "source.add",
       {{"id", "attn"}, {"catalog", "live:attn"}, {"transform", json::array({{{"op", "slice"}, {"axis", 0}, {"index", 0}}})}});
    ok(port, token, "panel.create",
       {{"id", "world"}, {"primitive", "scene"}, {"title", "Attention"},
        {"entity", json::array({{{"path", "attn"}, {"draw", "surface"}, {"bind", "attn"}}})}});
    ok(port, token, "instrument.save");
    ok(port, token, "instrument.open", {{"id", "pan"}});
    std::ofstream(t.str("spec-ready")).put('\n');

    // The first pan and orbit while the stream runs, the second after it ended.
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("streaming")); }, 90000), app.text);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
    CHECK(app_says(app, "[live] session ended", 20000));
    std::ofstream(t.str("ended")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("menu")); }, 90000), app.text);
    REQUIRE(script_ended(app, 30000));
    CHECK_MESSAGE(app.text.find("[ui] done") != std::string::npos, app.text);
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);

    const std::vector<FrameRow> rows = frame_rows(t.str("frames.tsv"));
    REQUIRE(rows.size() > 100);
    const auto pans = button_runs(rows, "world", 2), orbits = button_runs(rows, "world", 1);
    REQUIRE(pans.size() == 3);   // two pans, then the right click of the menu
    REQUIRE(orbits.size() == 2);
    for (size_t k = 0; k < 2; ++k) {
        CAPTURE(k);
        const auto [pan_moves, pan_camera] = moves_in(rows, pans[k]);
        CHECK(pan_moves == drag_steps);
        CHECK(pan_camera == pan_moves);   // the pan moves the camera at every move
        // No menu from the pan, from its release until the orbit.
        for (size_t i = pans[k].first; i < orbits[k].first; ++i) CHECK(rows[i].popup == 0);
        const auto [orbit_moves, orbit_camera] = moves_in(rows, orbits[k]);
        MESSAGE("orbit " << k << ": the camera moved at " << orbit_camera << " of " << orbit_moves << " moves");
        CHECK(orbit_moves == drag_steps);
        CHECK(orbit_camera == orbit_moves);
    }
    // A right click without a drag opens the panel's menu.
    bool opened = false;
    for (size_t i = pans[2].second; i < rows.size() && i < pans[2].second + 12; ++i) opened = opened || rows[i].popup == 1;
    CHECK(opened);
}

// ---- A drag whose press comes before the window saw the pointer move --------------------------
// GLFW's macOS backend gives a press without its place, and a window that got no cursor moves
// before it (one that was not the key window) knew the pointer only where it last saw it. The app
// gives ImGui the pointer's place with the press, so the first frame of an orbit or a pan moves
// the camera by about as much as the frames after it, not by the whole way from the stale place.
// The presses come through the platform's callback (the spec driver's `stale` drags).

namespace {

// The camera's numbers in the frame log's words: target x, y, z, azimuth and elevation.
std::vector<double> camera_numbers(const std::string& text) {
    std::vector<double> v;
    std::istringstream in(text);
    for (std::string w; in >> w;)
        if (w == "target") {
            double x = 0, y = 0, z = 0;
            in >> x >> y >> z;
            v.insert(v.end(), {x, y, z});
        } else if (w == "azimuth" || w == "elevation") {
            double a = 0;
            in >> a;
            v.push_back(a);
        }
    return v;
}

// The camera's steps over the frames of a run in which it moved, in order.
std::vector<double> camera_steps(const std::vector<FrameRow>& rows, std::pair<size_t, size_t> run) {
    std::vector<double> steps;
    for (size_t i = run.first + 1; i <= run.second + 1 && i < rows.size(); ++i) {
        const std::vector<double> a = camera_numbers(rows[i - 1].camera), b = camera_numbers(rows[i].camera);
        if (a.size() != 5 || b.size() != 5) continue;
        double d = 0;
        for (size_t k = 0; k < 5; ++k) d += (b[k] - a[k]) * (b[k] - a[k]);
        if (d > 0) steps.push_back(std::sqrt(d));
    }
    return steps;
}

}  // namespace

TEST_CASE("app live: an orbit and a pan whose press comes before the pointer's moves start without a jump") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("stale");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    Proc producer;
    const std::string link = start_producer(producer, env);
    // Each drag moves the pointer once a frame for this many frames (spec_ui_driver.h).
    const int drag_steps = 14;
    const std::string step_words = " steps " + std::to_string(drag_steps) + ";";
    const std::string script = "wait-file " + t.str("spec-ready") + ";page spec;wait 30;" +
                               "drag panel:world 0.25 0.5 to panel:world 0.75 0.6 stale" + step_words + "wait 10;" +
                               "drag panel:world 0.35 0.45 to panel:world 0.65 0.55 right stale" + step_words +
                               "wait 10;touch " + t.str("dragged");
    std::vector<std::string> e = app_env(env, t, script, shots);
    e.push_back("CALIPER_FRAME_LOG=" + t.str("frames.tsv"));
    Proc app;
    REQUIRE(app.start(env.exe,
                      {"--hidden", "--instruments", env.instruments, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      e));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        ::kill(producer.pid, SIGTERM);
        producer.wait(10000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE(token.size() >= 22);
    REQUIRE(eventually(app, [&] { return live_entries(port, token).count("live:attn") == 1; }, 60000));
    ok(port, token, "spec.new", {{"id", "stale"}, {"title", "Stale"}});
    ok(port, token, "source.add",
       {{"id", "attn"}, {"catalog", "live:attn"}, {"transform", json::array({{{"op", "slice"}, {"axis", 0}, {"index", 0}}})}});
    ok(port, token, "panel.create",
       {{"id", "world"}, {"primitive", "scene"}, {"title", "Attention"},
        {"entity", json::array({{{"path", "attn"}, {"draw", "surface"}, {"bind", "attn"}}})}});
    ok(port, token, "instrument.save");
    ok(port, token, "instrument.open", {{"id", "stale"}});
    std::ofstream(t.str("spec-ready")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("dragged")); }, 90000), app.text);
    REQUIRE(script_ended(app, 30000));
    CHECK_MESSAGE(app.text.find("[ui] done") != std::string::npos, app.text);
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);

    const std::vector<FrameRow> rows = frame_rows(t.str("frames.tsv"));
    REQUIRE(rows.size() > 100);
    const auto orbits = button_runs(rows, "world", 1), pans = button_runs(rows, "world", 2);
    REQUIRE(orbits.size() == 1);
    REQUIRE(pans.size() == 1);
    for (const auto& drag : {std::make_pair(std::string("orbit"), orbits[0]), std::make_pair(std::string("pan"), pans[0])}) {
        const std::string& name = drag.first;
        CAPTURE(name);
        std::vector<double> steps = camera_steps(rows, drag.second);
        // The camera moves at each of the pointer's moves.
        REQUIRE(steps.size() == (size_t)drag_steps);
        std::vector<double> sorted = steps;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2];
        double first = 0;
        for (size_t k = 0; k < 3; ++k) first = std::max(first, steps[k]);
        MESSAGE(name << ": the largest of the first three camera steps is " << first / median << " times the median");
        CHECK(first <= 3.0 * median);
    }
}

// ---- A slow start of an orbit and a pan, and the right click's tolerance ----------------------
// A scene panel's camera follows the pointer from the first point of motion after a press
// (live/camera_drag.h). At a point a frame, the camera moves on the pointer's first move, and no
// point is lost: the camera moves one step for each point. The release decides the click: a right
// press that wobbles a point puts the camera back and opens the panel's menu, as a right click does,
// and a right drag of 3 points, or one that goes 30 points out and comes back within a point, pans
// and opens no menu. On a panel without a scene a right click that slips 4 points still opens the
// menu, within ImGui's drag threshold as before. The scene panel is a tab, whose × lies over its
// picture: a click there removes the panel and moves no camera.

namespace {

// A drag's run of frames in the frame log: the frames from the pointer's first move to the camera's
// first move (-1 when the camera never moved), the camera's steps, and the pointer's distance from
// the press at the run's last frame, in points.
struct DragStart {
    int frames = -1;
    std::vector<double> steps;
    double pointer = 0;
};

DragStart drag_start(const std::vector<FrameRow>& rows, std::pair<size_t, size_t> run) {
    DragStart d;
    auto xy = [](const std::string& p) {
        double x = 0, y = 0;
        std::istringstream(p) >> x >> y;
        return std::make_pair(x, y);
    };
    size_t first_move = 0, first_camera = 0;
    for (size_t i = run.first + 1; i <= run.second + 1 && i < rows.size(); ++i) {
        if (!first_move && i <= run.second && rows[i].pointer != rows[run.first].pointer) first_move = i;
        if (!first_camera && rows[i].camera != rows[run.first].camera) first_camera = i;
    }
    if (first_move && first_camera) d.frames = int(first_camera) - int(first_move);
    d.steps = camera_steps(rows, run);
    const auto a = xy(rows[run.first].pointer), b = xy(rows[run.second].pointer);
    d.pointer = std::hypot(b.first - a.first, b.second - a.second);
    return d;
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

double sum_of(const std::vector<double>& v) {
    double s = 0;
    for (const double x : v) s += x;
    return s;
}

}  // namespace

TEST_CASE("app live: a slow orbit and pan of a scene panel move the camera on their first frame of motion, a right "
          "press that wobbles a point puts the camera back and opens the menu, and longer right drags pan without it") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("slow");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-instruments"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    Proc producer;
    const std::string link = start_producer(producer, env);
    // A point a frame: 30 moves of one point from 60 points right of the panel's left edge, at its
    // middle height; then right drags of 3 points, of 1 point, and of 30 points out and 29 back, a
    // right click, and a right click on the heatmap that slips 4 points.
    const std::string script = "wait-file " + t.str("spec-ready") +
                               ";page spec;wait 30;"
                               "drag panel:world 60 0.5 to panel:world 90 0.5 steps 30;wait 10;"
                               "drag panel:world 60 0.5 to panel:world 90 0.5 right steps 30;wait 10;"
                               "drag panel:world 60 0.5 to panel:world 63 0.5 right steps 3;wait 10;"
                               "drag panel:world 60 0.5 to panel:world 61 0.5 right steps 1;wait 10;key escape;wait 5;"
                               "drag panel:world 60 0.5 to panel:world 90 0.5 then panel:world 61 0.5 right steps 30;"
                               "wait 10;menu panel:world;wait 10;key escape;wait 5;"
                               "drag panel:heat 60 0.5 to panel:heat 64 0.5 right steps 4;wait 10;key escape;wait 5;"
                               "click remove:world;wait 10;touch " +
                               t.str("menu") + ";wait 5";
    std::vector<std::string> e = app_env(env, t, script, shots);
    e.push_back("CALIPER_FRAME_LOG=" + t.str("frames.tsv"));
    Proc app;
    REQUIRE(app.start(env.exe,
                      {"--hidden", "--instruments", env.instruments, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      e));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        ::kill(producer.pid, SIGTERM);
        producer.wait(10000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Instruments in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE(token.size() >= 22);
    REQUIRE(eventually(app, [&] { return live_entries(port, token).count("live:attn") == 1; }, 60000));
    ok(port, token, "spec.new", {{"id", "slow"}, {"title", "Slow"}});
    ok(port, token, "source.add",
       {{"id", "attn"}, {"catalog", "live:attn"}, {"transform", json::array({{{"op", "slice"}, {"axis", 0}, {"index", 0}}})}});
    ok(port, token, "panel.create",
       {{"id", "world"}, {"primitive", "scene"}, {"title", "Attention"},
        {"entity", json::array({{{"path", "attn"}, {"draw", "surface"}, {"bind", "attn"}}})}});
    ok(port, token, "panel.create", {{"id", "heat"}, {"primitive", "heatmap"}, {"title", "Map"}, {"bind", {{"values", "attn"}}}});
    ok(port, token, "panel.create", {{"id", "spare"}, {"primitive", "heatmap"}, {"title", "Spare"}, {"bind", {{"values", "attn"}}}});
    ok(port, token, "layout.set",
       {{"layout", json::parse(R"({"root":"main","nodes":[{"id":"main","split":"horizontal","ratio":0.6,"children":["side","heat"]},)"
                               R"({"id":"side","split":"tabs","children":["world","spare"]}]})")}});
    ok(port, token, "instrument.save");
    ok(port, token, "instrument.open", {{"id", "slow"}});
    std::ofstream(t.str("spec-ready")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("menu")); }, 90000), app.text);
    REQUIRE(script_ended(app, 30000));
    CHECK_MESSAGE(app.text.find("[ui] done") != std::string::npos, app.text);
    // The click on the tab's × removed the scene panel.
    const std::string shown = ok(port, token, "spec.get").dump();
    CHECK(shown.find("\"id\":\"world\"") == std::string::npos);
    CHECK(shown.find("\"id\":\"heat\"") != std::string::npos);
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);

    const std::vector<FrameRow> rows = frame_rows(t.str("frames.tsv"));
    REQUIRE(rows.size() > 100);
    const auto orbits = button_runs(rows, "world", 1), pans = button_runs(rows, "world", 2);
    REQUIRE(orbits.size() == 2);   // the slow orbit, and the click on the tab's ×
    // The scene never held the × click, and its camera stayed.
    for (size_t i = orbits[1].first; i <= orbits[1].second; ++i) {
        CAPTURE(i);
        CHECK(rows[i].panel_down == 0);
        CHECK(rows[i].camera == rows[orbits[1].first].camera);
    }
    // The slow pan, the right drags of 3 points, of 1 point and out and back, and the right click.
    REQUIRE(pans.size() == 5);
    double pan_step = 0;
    for (const auto& drag : {std::make_pair(std::string("orbit"), orbits[0]), std::make_pair(std::string("pan"), pans[0])}) {
        const std::string& name = drag.first;
        CAPTURE(name);
        const DragStart d = drag_start(rows, drag.second);
        MESSAGE(name << ": the camera first moved " << d.frames << " frames after the pointer's first move");
        CHECK(d.pointer == 30.0);
        CHECK(d.frames == 0);
        REQUIRE(d.steps.size() == 30);
        const double median = median_of(d.steps);
        // No jump, and no point lost: the camera moved one step for each point of the pointer's way.
        MESSAGE(name << ": the first camera step is " << d.steps[0] / median << " times the median; the camera "
                     << "moved " << sum_of(d.steps) / median << " median steps for 30 points");
        CHECK(d.steps[0] <= 3.0 * median);
        CHECK(sum_of(d.steps) == doctest::Approx(30.0 * median).epsilon(1e-3));
        if (name == "pan") pan_step = median;
    }
    auto popups = [&](size_t from, size_t to) {
        int n = 0;
        for (size_t i = from; i < to && i < rows.size(); ++i) n += rows[i].popup;
        return n;
    };
    auto distance = [](const std::string& a, const std::string& b) {
        const std::vector<double> x = camera_numbers(a), y = camera_numbers(b);
        double d = 0;
        for (size_t k = 0; k < x.size() && k < y.size(); ++k) d += (x[k] - y[k]) * (x[k] - y[k]);
        return std::sqrt(d);
    };
    // The right drag of 3 points pans 3 steps of the slow pan from its first point and opens no menu.
    const DragStart short_drag = drag_start(rows, pans[1]);
    CHECK(short_drag.pointer == 3.0);
    CHECK(short_drag.frames == 0);
    CHECK(sum_of(short_drag.steps) == doctest::Approx(3.0 * pan_step).epsilon(1e-3));
    CHECK(popups(pans[1].first, pans[2].first) == 0);
    // The right press that wobbles a point pans on that point, puts the camera back exactly at its
    // release and opens the menu.
    REQUIRE(pans[2].second + 1 < rows.size());
    const DragStart wobble = drag_start(rows, pans[2]);
    CHECK(wobble.pointer == 1.0);
    CHECK(wobble.frames == 0);
    CHECK(rows[pans[2].second + 1].camera == rows[pans[2].first].camera);
    CHECK(popups(pans[2].second, pans[2].second + 12) > 0);
    // Out 30 points and back within one: a drag, which keeps its camera and opens no menu.
    REQUIRE(pans[3].second + 1 < rows.size());
    CHECK(drag_start(rows, pans[3]).pointer == 1.0);
    const double kept = distance(rows[pans[3].second + 1].camera, rows[pans[3].first].camera);
    CHECK(kept == doctest::Approx(pan_step).epsilon(1e-3));   // one point of pan from the press
    CHECK(popups(pans[3].first, pans[4].first) == 0);
    // The right click without a drag leaves the camera and opens the menu.
    REQUIRE(pans[4].second + 1 < rows.size());
    CHECK(rows[pans[4].second + 1].camera == rows[pans[4].first].camera);
    CHECK(popups(pans[4].second, pans[4].second + 12) > 0);
    // The heatmap's right click that slipped 4 points opens the menu.
    const auto heat = button_runs(rows, "-", 2);
    REQUIRE(heat.size() == 1);
    CHECK(popups(heat[0].second, heat[0].second + 12) > 0);
}
