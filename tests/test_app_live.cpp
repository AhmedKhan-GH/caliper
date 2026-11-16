// The acceptance case "plot attention entropy per layer while it trains" in
// the hidden app (the hidden-app pattern of docs/explanations/testing.md).
// caliper_attention_producer (tests/live_attention_producer.cpp) stands in for
// a training script: a hub of its own and a producer that streams a model's
// attention as it sharpens. The app attaches to it with --url, and the test,
// through Caliper API's command endpoints alone, builds a spec over the live
// sources, opens it on the spec page, and checks that the frame and the
// heatmap panel change from one capture to the next while the stream runs,
// that the catalog shows the live sources with their shapes and rates, and
// that after the stream is detached (launching another applet takes it, as
// Home does) the spec opens again with its live panels offline and the app
// keeps running. POSIX with Metal only; without Metal the case is skipped.
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

}  // namespace

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

    // Detached: launching another applet takes the stream, as Home does.
    // Opened again, the spec's live panels are offline and the app runs on.
    CHECK(ok(port, token, "app.launch", {{"id", "dev.caliper.hello"}})["status"] == "ACTIVE");
    const auto gone_until = std::chrono::steady_clock::now() + 10s;
    while (!live_entries(port, token).empty() && std::chrono::steady_clock::now() < gone_until)
        std::this_thread::sleep_for(100ms);
    CHECK(live_entries(port, token).empty());
    CHECK(ok(port, token, "applet.open", {{"id", "attention-entropy"}})["title"] == "Attention entropy");
    const Image offline = decode(ok(port, token, "panel.render", {{"id", "heads"}, {"width", 480}, {"height", 320}}));
    CHECK(offline.px != heads_b.px);
    const Image page = decode(ok(port, token, "capture.frame"));
    CHECK(page.colours() > 1);

    ::kill(app.pid, SIGTERM);
    CHECK_MESSAGE(app.wait(30000) == 0, app.text);
    CHECK(app.text.find("[live] detached") != std::string::npos);
    CHECK(app.text.find("[session] command applet.open (api)") != std::string::npos);
    ::kill(producer.pid, SIGTERM);
    CHECK(producer.wait(10000) == 0);
}
