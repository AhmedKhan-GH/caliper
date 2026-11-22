// The whole route of an outside agent against the real app (the hidden-app
// pattern of docs/development/testing.md): `caliper --hidden --api
// --api-port 0 --workspace <dir>` runs as a child with an app-data folder of
// its own, and the test drives it through Caliper API's command endpoints
// alone, as a program or an agent does. It imports a Parquet file and a
// safetensors file, which runs their loaders, builds the three-panel dashboard
// of dashboard.spec.toml command by command, saves it, opens it, which shows
// the app's spec page, renders one panel with panel.render and captures the
// frame. Then caliper-mcp runs against the same app's API file and calls one
// command tool, and SIGTERM closes the app. A second case runs `caliper
// shot` on a workspace's spec applet that a command script builds. POSIX
// only; without a Metal device the cases say they are skipped.
#include <doctest/doctest.h>

#include "mcp_child.h"
#include "parquet_writer.h"
#include "test_proc.h"
#include "test_temp.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name) : path(caliper_test::unique_temp_path("caliper-route-" + name)) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str(const std::string& sub) const { return (path / sub).string(); }
};

// One HTTP/1.1 request to 127.0.0.1:<port>, answered on a connection of its
// own, as Caliper API serves one request per connection.
struct Http {
    int status = 0;
    std::string body;
};

Http http(uint16_t port, const std::string& method, const std::string& target, const std::string& token,
          const std::string& body = {}, const std::string& extra_headers = {}) {
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
    req += extra_headers + "\r\n" + body;
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

// The person's own token, from the app's start line: it is in no file.
std::string start_line_token(const std::string& text) {
    const std::string mark = "with the session token ";
    const size_t at = text.find(mark);
    if (at == std::string::npos) return {};
    const size_t s = at + mark.size();
    size_t e = s;
    while (e < text.size() && (std::isalnum((unsigned char)text[e]) || text[e] == '_' || text[e] == '-')) ++e;
    return text.substr(s, e - s);
}

// POST /v0/commands/{name} with `token`, whose origin the command runs from;
// the result envelope.
json command(uint16_t port, const std::string& token, const std::string& name, const json& args = json::object(),
             const std::string& extra_headers = {}) {
    const Http r = http(port, "POST", "/v0/commands/" + name, token, args.dump(), extra_headers);
    REQUIRE_MESSAGE(r.status == 200, name << ": " << r.status << " " << r.body);
    json env = json::parse(r.body, nullptr, false);
    REQUIRE(env.is_object());
    return env;
}

json ok(uint16_t port, const std::string& token, const std::string& name, const json& args = json::object()) {
    const json env = command(port, token, name, args);
    REQUIRE_MESSAGE(env["ok"] == true, name << ": " << env.dump());
    return env["result"];
}

size_t count_in(const std::string& text, const std::string& what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
    return n;
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
    std::vector<uint8_t> px;   // RGBA8
    size_t colours() const {
        std::set<uint32_t> c;
        for (size_t i = 0; i + 3 < px.size(); i += 4) {
            uint32_t v;
            std::memcpy(&v, &px[i], 4);
            c.insert(v);
            if (c.size() > 4096) break;
        }
        return c.size();
    }
};

Image decode(const json& result) {
    const std::string png = unb64(result["png_base64"].get<std::string>());
    Image img;
    int n = 0;
    unsigned char* px =
        stbi_load_from_memory((const unsigned char*)png.data(), (int)png.size(), &img.w, &img.h, &n, 4);
    REQUIRE_MESSAGE(px != nullptr, "not a PNG");
    img.px.assign(px, px + (size_t)img.w * img.h * 4);
    stbi_image_free(px);
    CHECK(img.w == result["width"].get<int>());
    CHECK(img.h == result["height"].get<int>());
    return img;
}

// A safetensors file of one f32 tensor [layers, heads, tokens, tokens] of
// causal attention: each head looks at the previous token, with a sink on the
// first.
void write_attention(const std::string& path) {
    const int L = 2, H = 4, T = 32;
    std::vector<float> w((size_t)L * H * T * T, 0.0f);
    for (int l = 0; l < L; ++l)
        for (int h = 0; h < H; ++h)
            for (int q = 0; q < T; ++q) {
                float* row = &w[(((size_t)l * H + h) * T + q) * T];
                double sum = 0;
                for (int k = 0; k <= q; ++k) {
                    const double s = (k == q - 1 ? 4.0 : 0.0) + (k == 0 ? 2.0 + h : 0.0);
                    row[k] = (float)std::exp(s);
                    sum += row[k];
                }
                for (int k = 0; k <= q; ++k) row[k] = (float)(row[k] / sum);
            }
    const json header = {{"weights", {{"dtype", "F32"}, {"shape", {L, H, T, T}}, {"data_offsets", {0, w.size() * 4}}}}};
    std::string h = header.dump();
    while (h.size() % 8) h += ' ';
    std::ofstream out(path, std::ios::binary);
    const uint64_t n = h.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out << h;
    out.write(reinterpret_cast<const char*>(w.data()), (std::streamsize)(w.size() * 4));
}

}  // namespace

TEST_CASE("app route: an agent builds, opens, renders and captures a dashboard in the hidden app over HTTP and MCP") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE is not set: run the ctest entry caliper_app_route");
        return;
    }
    TempDir t("app");
    const std::string ws = t.str("ws");
    const std::string data = t.str("data");
    // The files sit in the workspace, since an api call may not name a path
    // outside it.
    session_test::write_parquet(ws + "/data/run42_metrics.parquet",
                                "SELECT i AS step, 0.1 + 2.0 * exp(-i / 300.0) AS loss, "
                                "0.1 + 0.85 * (1 - exp(-i / 400.0)) AS accuracy FROM range(2000) t(i)");
    write_attention(ws + "/data/attn.safetensors");
    fs::create_directories(t.str("user-applets"));

    Proc p;
    REQUIRE(p.start(exe, {"--hidden", "--api", "--api-port", "0", "--workspace", ws},
                    {"CALIPER_DATA_DIR=" + data, "CALIPER_USER_APPLETS_DIR=" + t.str("user-applets"),
                     "CALIPER_WINDOW_SIZE=800x500"}));
    REQUIRE_MESSAGE(p.read_until("[api] Caliper API on ", 60000), p.text);
    REQUIRE_MESSAGE(p.read_until("[api] Spec applets in a browser", 10000), p.text);   // after the start line
    if (p.text.find("[renderer] metal") == std::string::npos) {
        const char* require = std::getenv("CALIPER_REQUIRE_GPU");
        REQUIRE_MESSAGE(!(require && std::string(require) == "1"), p.text);
        MESSAGE("caliper --hidden --api: run skipped: no GPU (no Metal device)");
        ::kill(p.pid, SIGTERM);
        p.wait(30000);
        return;
    }
    const json api = json::parse(std::ifstream(data + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    // The person's own scripting takes the token of the start line, whose
    // commands are not reviewed; the API file's is an outside agent's.
    const std::string token = start_line_token(p.text);
    REQUIRE(token.size() >= 22);
    REQUIRE(api["token"].get<std::string>() != token);

    // The command table, and the landing screen as the frame shows it.
    const Http table = http(port, "GET", "/v0/commands", token);
    REQUIRE_MESSAGE(table.status == 200, table.body << p.text);
    CHECK(json::parse(table.body)["commands"].size() > 30);
    const Image landing = decode(ok(port, token, "capture.frame"));
    CHECK(landing.colours() > 1);

    // Import both files, which runs their loaders.
    const json imported = ok(port, token, "data.import",
                             {{"paths", {"data/run42_metrics.parquet", "data/attn.safetensors"}}, {"run", true}});
    INFO(imported.dump());
    REQUIRE(imported["loaders"].size() == 2);
    for (const json& l : imported["loaders"]) {
        CHECK(l["ran"] == true);
        CHECK(l["outputs"].size() >= 1);
    }
    const json listed = ok(port, token, "catalog.list");
    INFO(listed.dump());
    std::set<std::string> catalog;
    for (const json& e : listed["entries"]) catalog.insert(e["id"].get<std::string>());
    CHECK(catalog.count("table:run42_metrics") == 1);
    CHECK(catalog.count("tensor:attn_weights") == 1);
    // The sources the app serves join the catalog once the session is open:
    // the host's feed channels.
    const json feeds = ok(port, token, "catalog.list", {{"kind", "feed"}});
    INFO(feeds.dump());
    CHECK(feeds["entries"].size() > 0);

    // The dashboard of dashboard.spec.toml, command by command.
    ok(port, token, "spec.new", {{"id", "run42-dashboard"}, {"title", "Run 42"}});
    ok(port, token, "source.add", {{"id", "metrics"}, {"catalog", "table:run42_metrics"}});
    ok(port, token, "source.add",
       {{"id", "attn"},
        {"catalog", "tensor:attn_weights"},
        {"transform",
         {{{"op", "slice"}, {"axis", 0}, {"index", 0}},
          {{"op", "reduce"}, {"axis", 0}, {"fn", "mean"}},
          {{"op", "normalise"}, {"mode", "minmax"}}}}});
    ok(port, token, "panel.create",
       {{"id", "curves"}, {"primitive", "series"}, {"title", "Loss and accuracy"},
        {"bind", {{"x", "metrics.step"}, {"y", {"metrics.loss", "metrics.accuracy"}}}},
        {"options", {{"x_label", "Step"}, {"y_log", true}}}});
    ok(port, token, "panel.create",
       {{"id", "attention"}, {"primitive", "heatmap"}, {"title", "Mean attention of layer 0"},
        {"bind", {{"values", "attn"}}},
        {"options", {{"colormap", "magma"}, {"range", {0.0, 1.0}}, {"x_label", "Key"}, {"y_label", "Query"}}}});
    ok(port, token, "panel.create",
       {{"id", "rows"}, {"primitive", "table"}, {"bind", {{"rows", "metrics"}}},
        {"options", {{"columns", {"step", "loss"}}}}});
    ok(port, token, "layout.set",
       {{"layout",
         {{"root", "main"},
          {"nodes",
           {{{"id", "main"}, {"split", "vertical"}, {"ratio", 0.65}, {"children", {"top", "rows"}}},
            {{"id", "top"}, {"split", "horizontal"}, {"children", {"curves", "attention"}}}}}}}});
    const json saved = ok(port, token, "applet.save");
    CHECK(saved["manifest"] == "applets/run42-dashboard/run42-dashboard.caliper.toml");

    // The app lists the saved applet at once, as a spec applet.
    std::string applets_body;
    INFO("GET /v0/applets: " << applets_body);
    auto applet_status = [&](const std::string& id) {
        const Http r = http(port, "GET", "/v0/applets", token);
        REQUIRE(r.status == 200);
        applets_body = r.body;
        const json listed = json::parse(r.body);   // kept alive through the loop
        for (const json& a : listed["applets"])
            if (a["id"] == id) return a["status"].get<std::string>();
        return std::string("absent");
    };
    CHECK(applet_status("local.run42-dashboard") == "READY");

    // Opening it shows the app's spec page, and the applet is ACTIVE while
    // the page shows it.
    CHECK(ok(port, token, "applet.open", {{"id", "run42-dashboard"}})["title"] == "Run 42");
    CHECK(applet_status("local.run42-dashboard") == "ACTIVE");
    // A launch of the applet that the page shows finds it open.
    const Http launched = http(port, "POST", "/v0/applets/local.run42-dashboard/launch", token);
    REQUIRE_MESSAGE(launched.status == 200, launched.body);
    CHECK(json::parse(launched.body)["status"] == "ACTIVE");

    // One panel, drawn off screen by the spec runtime.
    const Image panel = decode(ok(port, token, "panel.render", {{"id", "attention"}, {"width", 480}, {"height", 320}}));
    CHECK(panel.w == 480);
    CHECK(panel.h == 320);
    CHECK(panel.colours() > 16);

    // The frame now shows the dashboard: not blank, and not the landing.
    const Image page = decode(ok(port, token, "capture.frame"));
    CHECK(page.w == landing.w);
    CHECK(page.h == landing.h);
    CHECK(page.colours() > 16);
    CHECK(page.px != landing.px);

    // caliper-mcp against the same app: a command tool reads the open spec.
    {
        caliper_builder::mcp::test::McpChild child({"--api-file", data + "/api.json"});
        const json got = child.call("get", "spec_get", json::object());
        INFO(got.dump());
        CHECK(got["isError"] == false);
        CHECK(got["structuredContent"]["json"]["applet"]["id"] == "run42-dashboard");
        CHECK(child.finish().success());
        CHECK(child.all_err().find(token) == std::string::npos);
        CHECK(child.all_err().find(api["token"].get<std::string>()) == std::string::npos);
    }

    // SIGTERM closes the hidden window; the session closes and the API file
    // goes.
    ::kill(p.pid, SIGTERM);
    CHECK_MESSAGE(p.wait(30000) == 0, p.text);
    CHECK(p.text.find("[session] command applet.open (api)") != std::string::npos);
    CHECK(p.text.find("[session] workspace " + ws + " closed") != std::string::npos);
    CHECK_FALSE(fs::exists(data + "/api.json"));
    CHECK(fs::is_regular_file(ws + "/applets/run42-dashboard/run42-dashboard.spec.toml"));
}

TEST_CASE("app route: the API file's token is an outside agent's, mcp, so its spec edits are reviewed whatever the "
          "request claims, and the person's token is on the start line only") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE is not set: run the ctest entry caliper_app_route");
        return;
    }
    TempDir t("tokens");
    const std::string ws = t.str("ws");
    const std::string data = t.str("data");
    fs::create_directories(t.str("user-applets"));
    fs::create_directories(data);
    // The person denies the outside agent's two edits on the approval cards
    // (the hidden app's test decisions), so a held request answers at once.
    {
        std::ofstream d(t.str("decisions.txt"));
        d << "deny panel.create mcp\ndeny panel.create mcp\n";
    }
    Proc p;
    REQUIRE(p.start(exe, {"--hidden", "--api", "--api-port", "0", "--workspace", ws},
                    {"CALIPER_DATA_DIR=" + data, "CALIPER_USER_APPLETS_DIR=" + t.str("user-applets"),
                     "CALIPER_WINDOW_SIZE=640x400", "CALIPER_TEST_APPROVALS=" + t.str("decisions.txt")}));
    REQUIRE_MESSAGE(p.read_until("[api] Caliper API on ", 60000), p.text);
    REQUIRE_MESSAGE(p.read_until("[api] Spec applets in a browser", 10000), p.text);   // after the start line
    if (p.text.find("[renderer] metal") == std::string::npos) {
        MESSAGE("caliper --hidden --api: run skipped: no GPU (no Metal device)");
        ::kill(p.pid, SIGTERM);
        p.wait(30000);
        return;
    }
    const json api = json::parse(std::ifstream(data + "/api.json"), nullptr, false);
    REQUIRE(api.is_object());
    const uint16_t port = api["port"].get<uint16_t>();
    const std::string person = start_line_token(p.text);
    const std::string file_token = api["token"].get<std::string>();
    REQUIRE(person.size() >= 22);
    // The person's token is in no file Caliper writes.
    CHECK(file_token != person);
    for (const auto& e : fs::recursive_directory_iterator(data)) {
        if (!e.is_regular_file()) continue;
        std::ifstream in(e.path(), std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        CHECK_MESSAGE(ss.str().find(person) == std::string::npos, e.path().string());
    }
    // The person's scripting: unreviewed.
    ok(port, person, "spec.new", {{"id", "dash"}, {"title", "Dash"}});
    const json note = {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "Hi"}}}};
    // The API file's token, with no X-Caliper-Origin and with one that claims
    // api: both run as mcp, are held for review, and the person denies them.
    for (const std::string& claim : {std::string(), std::string("X-Caliper-Origin: api\r\n")}) {
        INFO("claim: " << claim);
        const json env = command(port, file_token, "panel.create", note, claim);
        INFO(env.dump());
        CHECK(env["ok"] == false);
        CHECK(env["error"]["code"] == "command.denied");
    }
    // The file's token reads as before.
    CHECK(http(port, "GET", "/v0/info", file_token).status == 200);
    // The person's own token: the same edit runs at once.
    CHECK(command(port, person, "panel.create", note)["ok"] == true);
    ::kill(p.pid, SIGTERM);
    CHECK_MESSAGE(p.wait(30000) == 0, p.text);
    CHECK(count_in(p.text, "panel.create (mcp) denied") == 2);
    CHECK(p.text.find("[session] command panel.create (api)") != std::string::npos);
}

TEST_CASE("app route: caliper shot draws a workspace's spec applet on the spec page, built by a command script") {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        MESSAGE("CALIPER_TEST_CALIPER_EXE is not set: run the ctest entry caliper_app_route");
        return;
    }
    TempDir t("shot");
    const std::string ws = t.str("ws");
    session_test::write_parquet(t.str("inputs/temps.parquet"),
                                "SELECT i AS step, 20 + 3 * sin(i / 20.0) AS temp FROM range(500) t(i)");
    // The script builds and saves the applet from the ui origin, which may
    // import a file outside the workspace; shot then opens it.
    const std::vector<json> script = {
        {{"name", "data.import"}, {"args", {{"paths", {t.str("inputs/temps.parquet")}}, {"run", true}}}},
        {{"name", "spec.new"}, {"args", {{"id", "temps-board"}, {"title", "Temperatures"}}}},
        {{"name", "source.add"}, {"args", {{"id", "t"}, {"catalog", "table:temps"}}}},
        {{"name", "panel.create"},
         {"args", {{"id", "curve"}, {"primitive", "series"}, {"bind", {{"x", "t.step"}, {"y", {"t.temp"}}}}}}},
        {{"name", "applet.save"}},
    };
    std::string text;
    for (const json& j : script) text += j.dump() + "\n";
    {
        std::ofstream out(t.str("build.jsonl"));
        out << text;
    }
    fs::create_directories(t.str("user-applets"));
    const std::vector<std::string> env = {"CALIPER_DATA_DIR=" + t.str("data"),
                                          "CALIPER_USER_APPLETS_DIR=" + t.str("user-applets")};

    Proc p;
    std::vector<std::string> env1 = env;
    env1.push_back("CALIPER_COMMAND_SCRIPT=" + t.str("build.jsonl"));
    REQUIRE(p.start(exe, {"shot", "--workspace", ws, "--launch", "temps-board", "--size", "640x400", "--frames", "5",
                          "--out", t.str("board.png")},
                    env1));
    const int rc = p.wait(120000);
    if (p.text.find("cannot draw offscreen") != std::string::npos) {
        MESSAGE("caliper shot: run skipped: no GPU");
        return;
    }
    REQUIRE_MESSAGE(rc == 0, p.text);
    CHECK(p.text.find("[session] script done: 5 commands, 0 failed") != std::string::npos);
    CHECK(p.text.find("[session] command applet.open (ui)") != std::string::npos);
    CHECK(p.text.find(": Temperatures at 640x400 after ") != std::string::npos);
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load(t.str("board.png").c_str(), &w, &h, &n, 4);
    REQUIRE(px != nullptr);
    Image img;
    img.w = w;
    img.h = h;
    img.px.assign(px, px + (size_t)w * h * 4);
    stbi_image_free(px);
    CHECK(w == 640);
    CHECK(h == 400);
    CHECK(img.colours() > 16);

    // The saved applet by its manifest id, which the loader lists from the
    // workspace's applets/ folder.
    Proc again;
    REQUIRE(again.start(exe, {"shot", "--workspace", ws, "--launch", "local.temps-board", "--size", "320x200",
                              "--frames", "3", "--out", t.str("again.png")},
                        env));
    REQUIRE_MESSAGE(again.wait(120000) == 0, again.text);
    CHECK(again.text.find(": Temperatures at 320x200 after ") != std::string::npos);
    CHECK(fs::is_regular_file(t.str("again.png")));

    // An id that neither the search folders nor the workspace has.
    Proc q;
    REQUIRE(q.start(exe, {"shot", "--workspace", ws, "--launch", "nowhere", "--size", "320x200", "--out",
                          t.str("none.png")},
                    env));
    CHECK(q.wait(120000) == 1);
    CHECK_MESSAGE(q.text.find("caliper: no applet with the id nowhere in the search folders or the workspace") !=
                      std::string::npos,
                  q.text);
    CHECK_FALSE(fs::exists(t.str("none.png")));
}
