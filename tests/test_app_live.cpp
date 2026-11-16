// The acceptance case "plot attention entropy per layer while it trains" in
// the hidden app (the hidden-app pattern of docs/explanations/testing.md).
// caliper_attention_producer (tests/live_attention_producer.cpp) stands in for
// a training script: a hub of its own and a producer that streams a model's
// attention as it sharpens. The app attaches to it with --url, and the test,
// through Caliper API's command endpoints alone, builds a spec over the live
// sources, opens it on the spec page, and checks that the frame and the
// heatmap panel change from one capture to the next while the stream runs,
// that the catalog shows the live sources with their shapes and rates, and
// that launching another applet keeps the stream attached. POSIX with Metal
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

#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
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
    std::string exe, producer, applets;
    std::string window, shots;   // CALIPER_TEST_WINDOW, CALIPER_TEST_SHOTS ("" when unset)
    bool ok = false;
};

Env test_env() {
    Env e;
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* producer = std::getenv("CALIPER_TEST_PRODUCER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !producer || !*producer || !applets || !*applets) return e;
    e.exe = exe;
    e.producer = producer;
    e.applets = applets;
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

// The app's environment: data, workspace's applets and window size, the
// input script and its shots folder.
std::vector<std::string> app_env(const Env& env, const TempDir& t, const std::string& script,
                                 const std::string& shots) {
    return {"CALIPER_DATA_DIR=" + t.str("data"), "CALIPER_USER_APPLETS_DIR=" + t.str("user-applets"),
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
        MESSAGE("CALIPER_TEST_CALIPER_EXE, CALIPER_TEST_PRODUCER_EXE or CALIPER_TEST_APPLETS is not set: run the "
                "ctest entry caliper_app_live_f4_stream");
        return;
    }
    TempDir t("f4");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-applets"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    Proc producer;
    const std::string link = start_producer(producer, env, t.str("data"));
    const std::string token = link.substr(link.find("token=") + 6);
    REQUIRE(token.size() >= 16);
    const std::string list_shot = shot_name(env, "f4-stream");
    const std::string live_shot = shot_name(env, "f4-stream-attached");

    // The person's steps: F4 on the landing, Enter on the listed stream.
    Proc app;
    REQUIRE(app.start(env.exe, {"--hidden", "--applets", env.applets, "--workspace", t.str("ws")},
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
    fs::create_directories(t.str("user-applets"));
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
                      {"--hidden", "--applets", env.applets, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      app_env(env, t, script, shots)));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Spec applets in a browser", 60000), app.text);
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
    ok(port, token, "applet.save");
    CHECK(ok(port, token, "applet.open", {{"id", "entropy"}})["title"] == "Entropy");
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
    CHECK(ok(port, token, "applet.open", {{"id", "entropy"}})["title"] == "Entropy");
    CHECK(renders_change());

    // F5 on the landing returns to the live page of the attached stream; ⌘D
    // Detach there ends it, and the spec's live panel goes offline.
    const Image last_live = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 320}, {"height", 200}}));
    std::ofstream(t.str("back")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("live2")); }, 60000), app.text);
    CHECK(app.text.find("[live] detached") == std::string::npos);
    std::ofstream(t.str("detach")).put('\n');
    REQUIRE_MESSAGE(eventually(app, [&] { return fs::exists(t.str("detached")); }, 60000), app.text);
    CHECK(app.text.find("[live] detached: attention_train.py") != std::string::npos);
    CHECK(eventually(app, [&] { return live_entries(port, token).empty(); }));
    CHECK(ok(port, token, "applet.open", {{"id", "entropy"}})["title"] == "Entropy");
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
    fs::create_directories(t.str("user-applets"));
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
                      {"--hidden", "--applets", env.applets, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      app_env(env, t, script, shots)));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Spec applets in a browser", 60000), app.text);
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

TEST_CASE("app live: files dropped on the live page start a new spec beside the live sources, or go into an open "
          "spec with unsaved changes") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("drop");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-applets"));
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
                      {"--hidden", "--applets", env.applets, "--url", link, "--api", "--api-port", "0", "--workspace",
                       t.str("ws")},
                      app_env(env, t, script, shots)));
    if (!has_metal(app)) {
        ::kill(app.pid, SIGTERM);
        app.wait(30000);
        return;
    }
    REQUIRE_MESSAGE(app.read_until("[api] Spec applets in a browser", 60000), app.text);
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
    ok(port, token, "applet.save");
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
        REQUIRE(app.start(env.exe, {"--hidden", "--applets", env.applets, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                          app_env(env, t, script, shots)));
        if (!has_metal(app)) {
            ::kill(app.pid, SIGTERM);
            app.wait(30000);
            return false;
        }
        REQUIRE_MESSAGE(app.read_until("[api] Spec applets in a browser", 60000), app.text);
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
    fs::create_directories(t.str("user-applets"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    const std::string shot = shot_name(env, "save-changes");
    const std::string home = "key escape;wait 5;key escape;page landing;";
    auto new_spec = [](const std::string& title) {
        return "key cmd+k;wait 5;type new " + title + ";wait 5;key enter;";
    };
    const std::string script = "page landing;wait 60;" + new_spec("Alpha") + "page spec;touch " + t.str("a1") + ";" + home +
                               new_spec("Beta") + "wait 30;touch " + t.str("a2") + ";save until;shot " + shot +
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
    CHECK(a.app.text.find("[save] Save changes to Alpha? waiting") != std::string::npos);
    // Cancel: nothing changed.
    a.reach(t, "a3");
    CHECK(a.spec()["id"] == "alpha");
    CHECK(a.spec()["dirty"] == true);
    // Don't Save: the new spec, and Alpha was never written.
    a.reach(t, "a4");
    CHECK(a.spec()["id"] == "beta");
    CHECK_FALSE(fs::exists(t.str("ws/applets/alpha")));
    // Save: Beta is saved, then the new spec.
    a.reach(t, "a5");
    CHECK(a.spec()["id"] == "gamma");
    CHECK(fs::exists(t.str("ws/applets/beta/beta.spec.toml")));
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
    fs::create_directories(t.str("user-applets"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    const std::string home = "key escape;wait 5;key escape;page landing;";
    const std::string open_two = "key cmd+k;wait 5;type Two;wait 5;key enter;";
    const std::string script = "wait-file " + t.str("b0") + ";page spec;" + home + open_two + "wait 30;touch " +
                               t.str("b1") + ";save save;page spec;wait 10;touch " + t.str("b2") + ";wait-file " +
                               t.str("b2go") + ";" + home + open_two + "page spec;wait 30;touch " + t.str("b3");
    SaveApp a;
    if (!a.start(env, t, script, shots)) return;
    ok(a.port, a.token, "spec.new", {{"id", "one"}, {"title", "One"}});
    ok(a.port, a.token, "applet.save");
    ok(a.port, a.token, "spec.new", {{"id", "two"}, {"title", "Two"}});
    ok(a.port, a.token, "applet.save");
    CHECK(ok(a.port, a.token, "applet.open", {{"id", "one"}})["title"] == "One");
    retitle(a, "One", "One edited");
    CHECK(a.spec()["dirty"] == true);
    go_on(t, "b0");

    // Two from the landing: the question first, and One stays open meanwhile.
    a.reach(t, "b1");
    CHECK(a.spec()["id"] == "one");
    CHECK(a.app.text.find("[save] Save changes to One edited? waiting") != std::string::npos);
    // Save: One is written with its change, then Two opens.
    a.reach(t, "b2");
    CHECK(a.spec()["id"] == "two");
    {
        std::ifstream f(t.str("ws/applets/one/one.spec.toml"));
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
    fs::create_directories(t.str("user-applets"));
    const std::string shots = env.shots.empty() ? t.str("") : env.shots;
    const std::string script = "wait-file " + t.str("c0") + ";close;wait 30;touch " + t.str("c1") +
                               ";save cancel;wait 30;touch " + t.str("c2") + ";wait-file " + t.str("c2go") +
                               ";close;save dont-save";
    SaveApp a;
    if (!a.start(env, t, script, shots)) return;
    ok(a.port, a.token, "spec.new", {{"id", "draft"}, {"title", "Draft"}});
    go_on(t, "c0");
    a.reach(t, "c1");
    CHECK(::kill(a.app.pid, 0) == 0);   // still running: the question holds the close back
    CHECK(a.app.text.find("[save] Save changes to Draft? waiting") != std::string::npos);
    a.reach(t, "c2");
    CHECK(::kill(a.app.pid, 0) == 0);   // Cancel: it runs on
    CHECK(a.spec()["id"] == "draft");
    go_on(t, "c2go");
    CHECK_MESSAGE(a.app.wait(60000) == 0, a.app.text);   // Don't Save: it quits
    CHECK(a.app.text.find("[save] Don't Save") != std::string::npos);
    CHECK_FALSE(fs::exists(t.str("ws/applets/draft")));
}

// ---- an agent's launch of a dashboard over unsaved changes ----

namespace {

// POST /v0/applets/<id>/launch with `token`.
Http launch_v0(uint16_t port, const std::string& token, const std::string& id) {
    return http(port, "POST", "/v0/applets/" + id + "/launch", token);
}

// The app lists the workspace's applet `id` (after an applet.save it rescans).
bool lists(SaveApp& a, const std::string& id) {
    return eventually(a.app, [&] {
        const Http r = http(a.port, "GET", "/v0/applets", a.token);
        return r.status == 200 && r.body.find("\"" + id + "\"") != std::string::npos;
    });
}

}  // namespace

TEST_CASE("app launch: an outside agent's launch of a dashboard over unsaved changes waits on an approval card; "
          "over a saved spec, and with the person's own token, it opens at once") {
    const Env env = test_env();
    if (!env.ok) return;
    TempDir t("launch-mcp");
    fs::create_directories(t.str("data"));
    fs::create_directories(t.str("user-applets"));
    const std::string decisions = t.str("decisions");
    std::ofstream(decisions).flush();
    SaveApp a;
    {
        std::vector<std::string> e = app_env(env, t, "wait 1", t.str(""));
        e.push_back("CALIPER_TEST_APPROVALS=" + decisions);
        REQUIRE(a.app.start(env.exe,
                            {"--hidden", "--applets", env.applets, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                            e));
        if (!has_metal(a.app)) {
            ::kill(a.app.pid, SIGTERM);
            a.app.wait(30000);
            return;
        }
        REQUIRE_MESSAGE(a.app.read_until("[api] Spec applets in a browser", 60000), a.app.text);
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
    ok(a.port, a.token, "applet.save");
    ok(a.port, a.token, "spec.new", {{"id", "two"}, {"title", "Two"}});
    ok(a.port, a.token, "applet.save");
    REQUIRE(lists(a, "local.one"));
    REQUIRE(lists(a, "local.two"));
    CHECK(ok(a.port, a.token, "applet.open", {{"id", "one"}})["title"] == "One");
    CHECK(a.spec()["dirty"] == false);
    auto cards = [&](const std::string& origin) { return a.count("applet.open (" + origin + ") waiting"); };

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
    std::ofstream(decisions, std::ios::app) << "approve applet.open mcp\n";
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
    std::ofstream(decisions, std::ios::app) << "deny applet.open mcp\n";
    CHECK(eventually(a.app, [&] { return a.app.text.find("applet.open (mcp) denied") != std::string::npos; }));
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
    fs::create_directories(t.str("user-applets"));
    const std::string decisions = t.str("decisions");
    std::ofstream(decisions).flush();
    SaveApp a;
    {
        std::vector<std::string> e = app_env(env, t, "wait 1", t.str(""));
        e.push_back("CALIPER_TEST_APPROVALS=" + decisions);
        REQUIRE(a.app.start(env.exe,
                            {"--hidden", "--applets", env.applets, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                            e));
        if (!has_metal(a.app)) {
            ::kill(a.app.pid, SIGTERM);
            a.app.wait(30000);
            return;
        }
        REQUIRE_MESSAGE(a.app.read_until("[api] Spec applets in a browser", 60000), a.app.text);
        a.port = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false)["port"].get<uint16_t>();
        a.token = start_token(a.app);
        REQUIRE(a.token.size() >= 22);
    }
    const std::string mcp = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false)["token"].get<std::string>();
    ok(a.port, a.token, "spec.new", {{"id", "one"}, {"title", "One"}});
    ok(a.port, a.token, "applet.save");
    ok(a.port, a.token, "spec.new", {{"id", "two"}, {"title", "Two"}});
    ok(a.port, a.token, "applet.save");
    REQUIRE(lists(a, "local.one"));
    REQUIRE(lists(a, "local.two"));
    CHECK(ok(a.port, a.token, "applet.open", {{"id", "one"}})["title"] == "One");

    // Not held (a saved spec): the answer is as before.
    {
        const Http r = launch_v0(a.port, mcp, "local.two");
        REQUIRE(r.status == 200);
        const json j = json::parse(r.body);
        CHECK(j["status"] == "ACTIVE");
        CHECK_FALSE(j.contains("note"));
    }
    CHECK(eventually(a.app, [&] { return a.spec()["id"] == "two"; }));

    // Held, through POST /v0/applets/{id}/launch: PENDING_APPROVAL with a note,
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
    REQUIRE_MESSAGE(eventually(a.app, [&] { return !(id = waiting_card(a.app, "applet.open", "mcp")).empty(); }),
                    a.app.text);
    {
        const Http r = http(a.port, "GET", "/v0/approvals/" + id, a.token);
        REQUIRE(r.status == 200);
        const json j = json::parse(r.body);
        CHECK(j["state"] == "pending");
        CHECK(j["name"] == "applet.open");
        CHECK(j["origin"] == "mcp");
    }
    CHECK(a.spec()["id"] == "two");
    std::ofstream(decisions, std::ios::app) << "approve applet.open mcp\n";
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
                        id2 = waiting_card(a.app, "applet.open", "mcp");
                        return !id2.empty() && id2 != id;
                    }),
                    a.app.text);
    std::ofstream(decisions, std::ios::app) << "deny applet.open mcp\n";
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
    fs::create_directories(t.str("user-applets"));
    // A first run saves the dashboard One into the workspace.
    {
        SaveApp first;
        if (!first.start(env, t, "wait 1", t.str(""))) return;
        ok(first.port, first.token, "spec.new", {{"id", "one"}, {"title", "One"}});
        ok(first.port, first.token, "applet.save");
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
    REQUIRE(app.start(env.exe, {"--hidden", "--applets", env.applets, "--api", "--api-port", "0", "--workspace", t.str("ws")},
                      e));
    REQUIRE_MESSAGE(app.read_until("[api] Spec applets in a browser", 60000), app.text);
    const json api = json::parse(std::ifstream(t.str("data") + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string token = start_token(app);
    REQUIRE_MESSAGE(app_says(app, "applet.open (agent) waiting", 60000), app.text);
    CHECK(ok(port, token, "spec.get")["id"] == "draft");
    std::ofstream(decisions, std::ios::app) << "approve applet.open agent\n";
    CHECK(eventually(app, [&] { return ok(port, token, "spec.get")["id"] == "one"; }));
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
}

TEST_CASE("app live: attention entropy per layer, from a live stream, on the spec page while it trains") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* producer_exe = std::getenv("CALIPER_TEST_PRODUCER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !producer_exe || !*producer_exe || !applets || !*applets) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE, CALIPER_TEST_PRODUCER_EXE or CALIPER_TEST_APPLETS is not set: run the "
                "ctest entry caliper_app_live_entropy");
        return;
    }
    const char* window = std::getenv("CALIPER_TEST_WINDOW");
    const char* shots = std::getenv("CALIPER_TEST_SHOTS");
    TempDir t("app");
    const std::string ws = t.str("ws");
    const std::string data = t.str("data");
    fs::create_directories(t.str("user-applets"));

    // The training script's stand-in: 6 layers of 8 heads over 64 tokens.
    Proc producer;
    REQUIRE(producer.start(producer_exe, {"--layers", "6", "--heads", "8", "--tokens", "64", "--hz", "8"}, {}));
    REQUIRE_MESSAGE(producer.read_until("[producer] hub ", 20000), producer.text);
    REQUIRE(producer.read_until("\n", 2000));
    const size_t at = producer.text.find("[producer] hub ") + std::string("[producer] hub ").size();
    const std::string link = producer.text.substr(at, producer.text.find('\n', at) - at);

    Proc app;
    REQUIRE(app.start(exe,
                      {"--hidden", "--applets", applets, "--url", link, "--api", "--api-port", "0", "--workspace", ws},
                      {"CALIPER_DATA_DIR=" + data, "CALIPER_USER_APPLETS_DIR=" + t.str("user-applets"),
                       std::string("CALIPER_WINDOW_SIZE=") + (window && *window ? window : "800x500")}));
    REQUIRE_MESSAGE(app.read_until("[api] Caliper API on ", 60000), app.text);
    REQUIRE_MESSAGE(app.read_until("[api] Spec applets in a browser", 10000), app.text);   // after the start line
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
    CHECK(live["live:attn"]["rate_hz"].get<double>() > 1.0);
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
    ok(port, token, "applet.save");
    CHECK(ok(port, token, "applet.open", {{"id", "attention-entropy"}})["title"] == "Attention entropy");

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

    // Launching another applet keeps the stream attached: only Detach on the
    // live page ends it. The catalog keeps the live sources, which keep
    // arriving, and the spec opened again draws live data.
    CHECK(ok(port, token, "app.launch", {{"id", "dev.caliper.hello"}})["status"] == "ACTIVE");
    CHECK(eventually(app, [&] {
        const auto e = live_entries(port, token);
        return e.count("live:attn.entropy") && e.at("live:attn.entropy").value("updated_at", std::string()) != updated_b;
    }));
    CHECK(app.text.find("[live] detached") == std::string::npos);
    CHECK(ok(port, token, "applet.open", {{"id", "attention-entropy"}})["title"] == "Attention entropy");
    const Image again = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 480}, {"height", 320}}));
    CHECK(again.colours() > 16);
    const Image page = decode(ok(port, token, "capture.frame"));
    CHECK(page.colours() > 1);

    // Quitting while attached detaches on the way out.
    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    CHECK(app.text.find("[live] detached") != std::string::npos);
    CHECK(app.text.find("[session] command applet.open (api)") != std::string::npos);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
}
