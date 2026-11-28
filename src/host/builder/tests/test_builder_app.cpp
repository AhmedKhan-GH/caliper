// Building by hand in the hidden app: `caliper --hidden` on an empty
// workspace, with the person's input played by CALIPER_TEST_SPEC_UI. A
// Parquet file and a safetensors file are dropped on the landing (the drop
// step calls the function GLFW's drop callback calls, since a hidden window
// receives no real drop), which makes a new spec and opens it in build mode
// with the two sources imported; the spec is named, the three-panel
// dashboard is built through the builder pane, the text strip and the side
// chat are opened in build mode, the instrument is saved with ⌘S, the page goes
// Home, and the dashboard is opened again from the landing's command line.
// The saved spec equals dashboard.spec.toml.
//
// It needs the caliper executable (CALIPER_TEST_CALIPER_EXE, set by its ctest
// entry) and a Metal device; without them it says it is skipped.
#include <stb_image.h>

#include "spec_test_workspace.h"

#include "metrics_store.h"
#include "open_url.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/journal.h>
#include <caliper_builder/spec/format.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace spec_test;
using nlohmann::json;

#ifndef _WIN32

namespace {

// The acceptance by hand, as one input script.
std::string acceptance_script(const std::string& data) {
    const std::string steps[] = {
        "wait 60",
        "drop " + data + "/run42_metrics.parquet " + data + "/attn.safetensors",
        "page spec",
        "pane source table:run42_metrics",
        "pane source tensor:attn_weights",
        "wait 20", "leave", "wait 5", "shot app-builder-1-dropped",
        // The spec's id, title and description, with nothing focused.
        "key escape",
        "click prop:spec.id", "type run42-dashboard", "key enter",
        "click prop:spec.title", "key cmd+a", "type Run 42", "key enter",
        "click prop:spec.description",
        "type Loss and accuracy of run 42, the attention of its first layer,\\nand the metrics as a table.",
        "click canvas 0.5 0.5",
        // The curves: the metrics table, named, as a series.
        "click source:table:run42_metrics 0.6 0.5",
        "click prop:source.id", "type metrics", "key enter",
        "click primitive:series", "pane focus series",
        "click prop:id", "type curves", "key enter", "pane focus curves",
        "click prop:title", "type Loss and accuracy", "key enter",
        "click option:x_label", "type Step", "key enter",
        "click option:y_log",
        "wait 10", "leave", "wait 10", "shot app-builder-2-series",
        // The attention: its source and three transforms, the heatmap beside the curves.
        "click source:tensor:attn_weights 0.6 0.5",
        "click prop:source.id", "type attn", "key enter",
        "click transform:add:slice", "click transform:add:reduce", "click transform:add:normalise",
        "drag primitive:heatmap to panel:curves -40 0.55", "pane focus heatmap",
        "click prop:id", "type attention", "key enter", "pane focus attention",
        "click prop:title", "type Mean attention of layer 0", "key enter",
        "click colormap:magma",
        "click option:range:min", "type 0", "key enter",
        "click option:x_label", "type Key", "key enter",
        "click option:y_label", "type Query", "key enter",
        // The table under everything, its columns, the root divider.
        "click source:spec:metrics 0.6 0.5",
        "drag primitive:table to canvas 0.5 -4", "pane focus table",
        "click prop:id", "type rows", "key enter", "pane focus rows",
        "click option:columns:accuracy",
        "drag divider: 0.5 0.0 to canvas 0.5 0.65",
        "wait 10", "leave", "wait 10", "shot app-builder-3-three-panels",
        // The spec text, then the side chat under Properties.
        "click bar:text", "click panel:curves 0.5 0.6", "wait 10", "leave", "wait 10", "shot app-builder-4-text",
        "chat open", "wait 40", "leave", "wait 10", "shot app-builder-5-with-chat", "chat close",
        // Saved, Home, and opened again from the landing's command line.
        "key cmd+s", "wait 20",
        "key escape", "wait 5", "key escape",
        "page landing", "wait 30",
        "key cmd+k", "wait 5", "type Run 42", "wait 5", "key enter",
        "page spec", "wait 40", "leave", "wait 10", "shot app-builder-6-reopened",
        "quit"};
    std::string out;
    for (const auto& s : steps) out += (out.empty() ? "" : "; ") + s;
    return out;
}

// One HTTP request to 127.0.0.1:<port>; the status and the body, 0 when
// nothing answered.
struct Answer {
    int status = 0;
    std::string body;
};
Answer http(int port, const std::string& method, const std::string& target, const std::string& extra = {},
            const std::string& body = {}) {
    Answer a;
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return a;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    timeval tv{10, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (::connect(fd, (sockaddr*)&addr, sizeof addr) != 0) {
        ::close(fd);
        return a;
    }
    const std::string req = method + " " + target + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                            "\r\nConnection: close\r\n" + extra + "\r\n" + body;
    (void)::send(fd, req.data(), req.size(), 0);
    std::string raw;
    char buf[65536];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        raw.append(buf, (size_t)n);
    }
    ::close(fd);
    const size_t end = raw.find("\r\n\r\n");
    if (raw.size() < 12 || end == std::string::npos) return a;
    a.status = std::atoi(raw.substr(9, 3).c_str());
    a.body = raw.substr(end + 4);
    return a;
}

// The value of `key` in a URL's query, "" when it has none.
std::string query_value(const std::string& url, const std::string& key) {
    const size_t q = url.find('?');
    if (q == std::string::npos) return {};
    std::string rest = url.substr(q + 1);
    while (!rest.empty()) {
        const size_t amp = rest.find('&');
        const std::string part = rest.substr(0, amp);
        if (part.rfind(key + "=", 0) == 0) return part.substr(key.size() + 1);
        if (amp == std::string::npos) break;
        rest = rest.substr(amp + 1);
    }
    return {};
}

// The token after `mark` in `text`, as the API's start line prints it.
std::string token_after(const std::string& text, const std::string& mark) {
    const size_t at = text.find(mark);
    if (at == std::string::npos) return {};
    size_t e = at + mark.size();
    while (e < text.size() && (std::isalnum((unsigned char)text[e]) || text[e] == '_' || text[e] == '-')) ++e;
    return text.substr(at + mark.size(), e - at - mark.size());
}

// A workspace with a saved dashboard, run42-dashboard, of one series panel
// over a CSV table, built by a command script that opens it at the start.
void saved_dashboard(const std::string& root) {
    std::filesystem::create_directories(root + "/ws/data");
    {
        std::ofstream csv(root + "/ws/data/run42_metrics.csv");
        csv << "step,loss,accuracy\n";
        for (int i = 0; i < 100; ++i) csv << i << "," << 2.0 / (1 + i) << "," << i / 100.0 << "\n";
    }
    std::ofstream f(root + "/build.jsonl");
    f << R"({"name":"data.import","args":{"paths":["data/run42_metrics.csv"],"run":true}})" << "\n"
      << R"({"name":"spec.new","args":{"id":"run42-dashboard","title":"Run 42"}})" << "\n"
      << R"({"name":"source.add","args":{"id":"metrics","catalog":"table:run42_metrics"}})" << "\n"
      << R"({"name":"panel.create","args":{"id":"curves","primitive":"series","bind":{"x":"metrics.step","y":["metrics.loss"]}}})"
      << "\n"
      << R"({"name":"instrument.save","args":{"overwrite":true}})" << "\n"
      << R"({"name":"instrument.open","args":{"id":"run42-dashboard"}})" << "\n";
}

// One run of the hidden app on root/ws, with root/data and root/apps, the
// switch every sidecar needs, and `env` and `args` added.
caliper_builder::acp::RunResult run_app(const std::string& exe, const std::string& root, const std::string& script,
                                        std::vector<caliper_builder::acp::EnvVar> env = {},
                                        std::vector<std::string> args = {}) {
    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--workspace", root + "/ws"};
    for (auto& a : args) o.argv.push_back(a);
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_INSTRUMENTS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "1440x900"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", script},
             {"CALIPER_TEST_SPEC_UI_SHOTS", root + "/shots"},
             {"CALIPER_EXIT_AFTER", "150"}};
    // A variable of `env` replaces the default of the same name.
    for (auto& e : env) {
        auto same = std::find_if(o.env.begin(), o.env.end(), [&](const auto& d) { return d.name == e.name; });
        if (same != o.env.end()) *same = e;
        else o.env.push_back(e);
    }
    o.env_remove = {"CALIPER_API"};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 200000;
    lim.max_output_bytes = 16u << 20;
    return caliper_builder::acp::run_process(o, lim);
}

// The test's root folder with the folders run_app() uses; "" when the case
// cannot run here (it says why).
std::string app_root(const char* exe, const std::string& name) {
    if (!exe || !*exe) {
        std::printf("caliper_builder_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return {};
    }
    const std::string root = fresh_dir(name);
    for (const auto& d : {root + "/shots", root + "/data", root + "/apps", root + "/ws"})
        std::filesystem::create_directories(d);
    return root;
}

}  // namespace

TEST_CASE("builder app: the dashboard built by hand in the hidden app, saved and opened again" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_builder_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("builder-app");
    const std::string ws = root + "/ws", data = root + "/files", shots = root + "/shots";
    for (const auto& d : {shots, root + "/data", root + "/apps"}) std::filesystem::create_directories(d);
    write_parquet(data + "/run42_metrics.parquet",
                  "SELECT range AS step, 0.1 + 2.0 * exp(-range / 50.0) AS loss, "
                  "0.1 + 0.8 * (1 - exp(-range / 40.0)) AS accuracy FROM range(200)");
    std::vector<float> w(2 * 3 * 8 * 8, 0.0f);
    for (int l = 0; l < 2; ++l)
        for (int h = 0; h < 3; ++h)
            for (int q = 0; q < 8; ++q) w[((l * 3 + h) * 8 + q) * 8 + q] = 1.0f + (float)h;
    write_safetensors_f32(data + "/attn.safetensors", "weights", {2, 3, 8, 8}, w);

    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--workspace", ws};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_INSTRUMENTS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "1440x900"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", acceptance_script(data)},
             {"CALIPER_TEST_SPEC_UI_SHOTS", shots},
             {"CALIPER_EXIT_AFTER", "200"}};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 300000;
    lim.max_output_bytes = 64u << 20;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 8000 ? run.err.size() - 8000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);

    // The saved instrument is the fixture.
    const std::string saved = ws + "/instruments/run42-dashboard/run42-dashboard.spec.toml";
    REQUIRE(std::filesystem::is_regular_file(saved));
    auto parsed = caliper_builder::spec::parse_toml(read_text(saved));
    REQUIRE(parsed.ok());
    CHECK(parsed.spec == dashboard_fixture());

    // The journal: the new spec and the import from the person's ui, the
    // build from the builder, the save, and the instrument opened again after it.
    const auto journal = caliper_builder::read_journal(ws + "/journal/commands.jsonl");
    bool made = false, imported = false, saved_line = false, reopened = false;
    size_t built = 0;
    for (const auto& e : journal.events) {
        if (e.name == "spec.new") made = made || e.origin == caliper_builder::CommandOrigin::ui;
        if (e.name == "data.import") imported = imported || (e.ok && e.origin == caliper_builder::CommandOrigin::ui);
        if (e.origin == caliper_builder::CommandOrigin::builder && e.ok) ++built;
        if (e.name == "instrument.save" && e.ok) saved_line = true;
        if (e.name == "instrument.open" && e.ok && saved_line) reopened = true;
    }
    CHECK(made);
    CHECK(imported);
    CHECK(built > 20);
    CHECK(saved_line);
    CHECK(reopened);

    // The six captures, at the window's size in points or pixels.
    for (const char* name : {"app-builder-1-dropped", "app-builder-2-series", "app-builder-3-three-panels",
                             "app-builder-4-text", "app-builder-5-with-chat", "app-builder-6-reopened"}) {
        int iw = 0, ih = 0, n = 0;
        const std::string file = shots + "/" + name + ".png";
        unsigned char* px = stbi_load(file.c_str(), &iw, &ih, &n, 4);
        CHECK_MESSAGE(px != nullptr, file);
        if (px) stbi_image_free(px);
        CHECK(((iw == 1440 && ih == 900) || (iw == 2880 && ih == 1800)));
    }
}

// New dashboard takes the landing's command line as its title, so the line is
// empty when the person comes back Home: with F3, and with "new <title>" and
// Enter. The line used to keep the title while it had the keyboard, since the
// line's own text won over the cleared buffer, and the instruments came back
// filtered by it.
TEST_CASE("builder app: New dashboard leaves the landing's command line empty" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_builder_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("builder-app-new");
    for (const auto& d : {root + "/data", root + "/apps"}) std::filesystem::create_directories(d);
    const std::string script =
        "wait 60; page landing; "
        "key cmd+k; type Run 42; key f3; page spec; wait 10; key escape; page landing; wait 10; command-line none; "
        // Run 42 was never saved, so the second New dashboard asks first.
        "key cmd+k; type new Run 43; key enter; save dont-save; page spec; wait 10; key escape; page landing; wait 10; "
        "command-line none; "
        "quit";
    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--workspace", root + "/ws"};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_INSTRUMENTS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "1440x900"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", script},
             {"CALIPER_EXIT_AFTER", "120"}};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 180000;
    lim.max_output_bytes = 16u << 20;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
    // Both dashboards were made from the person's origin, with their titles.
    std::vector<std::string> titles;
    for (const auto& e : caliper_builder::read_journal(root + "/ws/journal/commands.jsonl").events)
        if (e.name == "spec.new" && e.ok && e.origin == caliper_builder::CommandOrigin::ui)
            titles.push_back(e.args.value("title", std::string()));
    CHECK(titles == std::vector<std::string>{"Run 42", "Run 43"});
}

// Metrics in Sources, grouped by run, in the app's own data folder: twelve
// runs in its metrics store, a new dashboard, the newest five runs closed
// with Show all, the newest run opened and its loss made a series panel.
// CALIPER_TEST_WINDOW sets the window size (1440x900) and CALIPER_TEST_SHOTS
// a folder for the captures, metric-runs-<collapsed|expanded>-<W>x<H>.png.
TEST_CASE("builder app: a panel from a metric of a run opened in Sources" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-metric-runs");
    if (root.empty()) return;
    std::string window = "1440x900", shots = root + "/shots";
    if (const char* w = std::getenv("CALIPER_TEST_WINDOW"); w && *w) window = w;
    if (const char* d = std::getenv("CALIPER_TEST_SHOTS"); d && *d) shots = d;
    std::uint64_t newest = 0;
    {
        caliper_host::MetricsStore store;
        REQUIRE(store.open(root + "/data/metrics.duckdb"));
        for (int i = 1; i <= 12; ++i) {
            newest = store.begin_run("sweep", "lr-" + std::to_string(i));
            for (int step = 0; step < 60; ++step) {
                store.scalar(newest, "loss", step, 2.0 / (1.0 + 0.05 * i * step));
                store.scalar(newest, "accuracy", step, 1.0 - 1.0 / (1.0 + 0.04 * i * step));
            }
            store.end_run(newest);
        }
        store.close();
    }
    const std::string run = std::to_string(newest), metric = "metric:" + run + "/loss";
    const std::string script = "wait 60; page landing; key f3; page spec; pane source " + metric +
                               "; wait 10; leave; wait 5; shot metric-runs-collapsed-" + window +
                               "; click run:" + run + "; click source:" + metric +
                               " 0.6 0.5; click primitive:series; pane focus series; wait 20; leave; wait 10; "
                               "shot metric-runs-expanded-" + window + "; quit";
    const auto result = run_app(exe, root, script, {{"CALIPER_WINDOW_SIZE", window}, {"CALIPER_TEST_SPEC_UI_SHOTS", shots}});
    INFO("stderr tail: " << result.err.substr(result.err.size() > 4000 ? result.err.size() - 4000 : 0));
    REQUIRE(result.error.ok());
    if (result.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(result.status.success());
    CHECK(result.err.find("[ui] done") != std::string::npos);
    CHECK(std::filesystem::is_regular_file(shots + "/metric-runs-expanded-" + window + ".png"));
    // The pane added the metric as a source and a series panel over it.
    bool added = false, panel = false;
    for (const auto& e : caliper_builder::read_journal(root + "/ws/journal/commands.jsonl").events) {
        if (e.name == "source.add" && e.ok && e.args.value("catalog", std::string()) == metric) added = true;
        if (e.name == "panel.create" && e.ok && e.args.value("primitive", std::string()) == "series") panel = true;
    }
    CHECK(added);
    CHECK(panel);
}

// Pieces in Sources: two pieces in the workspace's pieces folder, a new
// dashboard in build mode, and the row of one selected in the group Pieces.
// CALIPER_TEST_WINDOW sets the window size (1440x900) and CALIPER_TEST_SHOTS
// a folder for the capture, pieces-<W>x<H>.png.
TEST_CASE("builder app: the workspace's pieces in Sources" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-pieces");
    if (root.empty()) return;
    std::string window = "1440x900", shots = root + "/shots";
    if (const char* w = std::getenv("CALIPER_TEST_WINDOW"); w && *w) window = w;
    if (const char* d = std::getenv("CALIPER_TEST_SHOTS"); d && *d) shots = d;
    auto piece = [&](const std::string& name, const std::string& manifest, const std::string& draw) {
        const std::string dir = root + "/ws/pieces/" + name + "/1";
        std::filesystem::create_directories(dir);
        std::ofstream(dir + "/piece.toml") << manifest;
        std::ofstream(dir + "/draw.wgsl") << draw;
    };
    piece("glow",
          "name = \"glow\"\nversion = 1\ndescription = \"A soft glow over the panel\"\n\n"
          "[draw]\ncamera = \"none\"\npasses = [{ kind = \"render\", fragment = \"shade\" }]\n",
          "@fragment fn shade() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.6, 1.0, 1.0); }\n");
    piece("rings",
          "name = \"rings\"\nversion = 1\ndescription = \"Rings around the selected item\"\n\n"
          "[draw]\ncamera = \"orbit\"\npasses = [{ kind = \"render\", fragment = \"ring\" }]\n",
          "@fragment fn ring() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.8, 0.2, 1.0); }\n");
    const std::string key = "piece:workspace/glow@1";
    const std::string script = "wait 60; page landing; key f3; page spec; pane source " + key + "; click source:" +
                               key + " 0.3 0.5; wait 10; leave; wait 5; shot pieces-" + window + "; quit";
    const auto result = run_app(exe, root, script, {{"CALIPER_WINDOW_SIZE", window}, {"CALIPER_TEST_SPEC_UI_SHOTS", shots}});
    INFO("stderr tail: " << result.err.substr(result.err.size() > 4000 ? result.err.size() - 4000 : 0));
    REQUIRE(result.error.ok());
    if (result.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(result.status.success());
    CHECK(result.err.find("[ui] done") != std::string::npos);
    CHECK(std::filesystem::is_regular_file(shots + "/pieces-" + window + ".png"));
}

// Removing a panel from the dashboard page, outside build mode: the × on its
// strip, Remove panel in its menu and ⌫ on the focused panel, each a
// panel.remove from the ui origin that ⌘Z takes back. The dashboard has a
// panel beside two tabs. CALIPER_TEST_WINDOW and CALIPER_TEST_SHOTS as above;
// the capture is panel-remove-hover-<W>x<H>.png, the × on the hovered panel.
TEST_CASE("builder app: a panel removed with its x, its menu and the delete key, and brought back with undo" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-remove-panel");
    if (root.empty()) return;
    std::string window = "1440x900", shots = root + "/shots";
    if (const char* w = std::getenv("CALIPER_TEST_WINDOW"); w && *w) window = w;
    if (const char* d = std::getenv("CALIPER_TEST_SHOTS"); d && *d) shots = d;
    std::filesystem::create_directories(root + "/ws/data");
    {
        std::ofstream csv(root + "/ws/data/run42_metrics.csv");
        csv << "step,loss,accuracy\n";
        for (int i = 0; i < 100; ++i) csv << i << "," << 2.0 / (1 + i) << "," << i / 100.0 << "\n";
    }
    {
        std::ofstream f(root + "/build.jsonl");
        f << R"({"name":"data.import","args":{"paths":["data/run42_metrics.csv"],"run":true}})" << "\n"
          << R"({"name":"spec.new","args":{"id":"run42-dashboard","title":"Run 42"}})" << "\n"
          << R"({"name":"source.add","args":{"id":"metrics","catalog":"table:run42_metrics"}})" << "\n"
          << R"({"name":"panel.create","args":{"id":"curves","primitive":"series","bind":{"x":"metrics.step","y":["metrics.loss"]}}})"
          << "\n"
          << R"({"name":"panel.create","args":{"id":"accuracy","primitive":"series","bind":{"x":"metrics.step","y":["metrics.accuracy"]}}})"
          << "\n"
          << R"({"name":"panel.create","args":{"id":"rows","primitive":"table","bind":{"rows":"metrics"}}})" << "\n"
          << R"({"name":"layout.set","args":{"layout":{"root":"main","nodes":[)"
          << R"({"id":"main","split":"horizontal","ratio":0.6,"children":["curves","side"]},)"
          << R"({"id":"side","split":"tabs","children":["accuracy","rows"]}]}}})" << "\n"
          << R"({"name":"instrument.save","args":{"overwrite":true}})" << "\n"
          << R"({"name":"instrument.open","args":{"id":"run42-dashboard"}})" << "\n";
    }
    const std::string script =
        "wait 60; page spec; wait 30; focus curves; wait 10; shot panel-remove-hover-" + window +
        "; click remove:curves; wait 10; key cmd+z; wait 10; "
        "menu panel:accuracy; wait 5; click menu:remove; wait 10; key cmd+z; wait 10; "
        "focus curves; key backspace; wait 10; key cmd+z; wait 10; leave; wait 5; quit";
    const auto result = run_app(exe, root, script,
                                {{"CALIPER_COMMAND_SCRIPT", root + "/build.jsonl"}, {"CALIPER_WINDOW_SIZE", window},
                                 {"CALIPER_TEST_SPEC_UI_SHOTS", shots}});
    INFO("stderr tail: " << result.err.substr(result.err.size() > 4000 ? result.err.size() - 4000 : 0));
    REQUIRE(result.error.ok());
    if (result.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(result.status.success());
    CHECK(result.err.find("[ui] done") != std::string::npos);
    CHECK(std::filesystem::is_regular_file(shots + "/panel-remove-hover-" + window + ".png"));
    // Three removals from the person's ui origin, each taken back by ⌘Z.
    std::vector<std::string> removed;
    size_t undone = 0;
    for (const auto& e : caliper_builder::read_journal(root + "/ws/journal/commands.jsonl").events) {
        if (e.name == "panel.remove" && e.event == "do" && e.ok) {
            CHECK(e.origin == caliper_builder::CommandOrigin::ui);
            removed.push_back(e.args.value("id", std::string()));
        }
        if (e.event == "undo" && e.ok) ++undone;
    }
    CHECK(removed == std::vector<std::string>{"curves", "accuracy", "curves"});
    CHECK(undone == 3);
}

TEST_CASE("builder app: non-closable panel and Lock Layout hide close controls and refuse removal" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-lock-closable");
    if (root.empty()) return;
    std::string window = "1440x900", shots = root + "/shots";
    if (const char* w = std::getenv("CALIPER_TEST_WINDOW"); w && *w) window = w;
    if (const char* d = std::getenv("CALIPER_TEST_SHOTS"); d && *d) shots = d;
    std::filesystem::create_directories(root + "/ws/data");
    {
        std::ofstream csv(root + "/ws/data/run42_metrics.csv");
        csv << "step,loss\n";
        for (int i = 0; i < 40; ++i) csv << i << "," << 1.0 / (1 + i) << "\n";
    }
    {
        std::ofstream f(root + "/build.jsonl");
        f << R"({"name":"data.import","args":{"paths":["data/run42_metrics.csv"],"run":true}})" << "\n"
          << R"({"name":"spec.new","args":{"id":"lock-test","title":"Lock Test"}})" << "\n"
          << R"({"name":"source.add","args":{"id":"metrics","catalog":"table:run42_metrics"}})" << "\n"
          << R"({"name":"panel.create","args":{"id":"curves","primitive":"series","closable":false,"bind":{"x":"metrics.step","y":["metrics.loss"]}}})"
          << "\n"
          << R"({"name":"panel.create","args":{"id":"rows","primitive":"table","bind":{"rows":"metrics"}}})" << "\n"
          << R"({"name":"instrument.save","args":{"overwrite":true}})" << "\n"
          << R"({"name":"instrument.open","args":{"id":"lock-test"}})" << "\n";
    }
    // 1) Non-closable `curves` ignores Delete/Backspace and exposes no remove:curves.
    // 2) Clicking Lock Layout locks the instrument so even closable `rows` ignores Delete/Backspace.
    // 3) Clicking Lock Layout again unlocks so `remove:rows` works and undo restores it.
    const std::string script =
        "wait 60; page spec; wait 30; focus curves; key backspace; wait 10; "
        "click bar:lock; wait 10; focus rows; key backspace; wait 10; "
        "click bar:lock; wait 10; click remove:rows; wait 10; key cmd+z; wait 10; leave; wait 5; quit";
    const auto result = run_app(exe, root, script,
                                {{"CALIPER_COMMAND_SCRIPT", root + "/build.jsonl"}, {"CALIPER_WINDOW_SIZE", window},
                                 {"CALIPER_TEST_SPEC_UI_SHOTS", shots}});
    INFO("stderr tail: " << result.err.substr(result.err.size() > 4000 ? result.err.size() - 4000 : 0));
    REQUIRE(result.error.ok());
    if (result.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(result.status.success());
    CHECK(result.err.find("[ui] done") != std::string::npos);
    std::vector<std::string> removed;
    std::vector<bool> lock_states;
    for (const auto& e : caliper_builder::read_journal(root + "/ws/journal/commands.jsonl").events) {
        if (e.name == "panel.remove" && e.event == "do" && e.ok)
            removed.push_back(e.args.value("id", std::string()));
        if (e.name == "layout.lock" && e.event == "do" && e.ok)
            lock_states.push_back(e.args.value("locked", false));
    }
    CHECK(lock_states == std::vector<bool>{true, false});
    CHECK(removed == std::vector<std::string>{"rows"});
}

// Open in browser hands the browser only a link of the viewer on this
// machine's loopback API, in the characters such a link holds.
TEST_CASE("builder app: only a link of the browser viewer on loopback goes to the browser") {
    using caliper_host::viewer_link_openable;
    CHECK(viewer_link_openable("http://127.0.0.1:7341/v0/spec?token=abc_DEF-123&instrument=training-run"));
    CHECK_FALSE(viewer_link_openable("http://127.0.0.1:7341/v0/info?token=abc"));
    CHECK_FALSE(viewer_link_openable("http://localhost:7341/v0/spec?token=abc"));
    CHECK_FALSE(viewer_link_openable("https://127.0.0.1:7341/v0/spec?token=abc"));
    CHECK_FALSE(viewer_link_openable("http://127.0.0.1:/v0/spec?token=abc"));
    CHECK_FALSE(viewer_link_openable("http://127.0.0.1:123456/v0/spec?token=abc"));
    CHECK_FALSE(viewer_link_openable("http://127.0.0.1:7341/v0/spec?token=abc&instrument=x;rm"));
    CHECK_FALSE(viewer_link_openable("http://127.0.0.1:7341/v0/spec?token=a b"));
    std::string why;
    CHECK_FALSE(caliper_host::open_in_default_browser("file:///etc/hosts", &why));
    CHECK(why == "not a link of the browser viewer");
    // The instrument a link names is a workspace's instrument id: a lower-case letter,
    // then lower-case letters, digits, _ or -, 64 at most; never . or ..
    using caliper_host::viewer_instrument_id;
    CHECK(viewer_instrument_id("training-run"));
    CHECK(viewer_instrument_id("a_1"));
    CHECK(viewer_instrument_id(std::string(64, 'a')));
    const std::vector<std::string> bad_ids = {".", "..", "", "Run", "1run", "-run", "run.42", "run/42",
                                              std::string(65, 'a')};
    for (const std::string& bad : bad_ids) {
        CHECK_FALSE_MESSAGE(viewer_instrument_id(bad), bad);
        CHECK_FALSE_MESSAGE(viewer_link_openable("http://127.0.0.1:7341/v0/spec?token=abc&instrument=" + bad), bad);
    }
    CHECK_FALSE(viewer_link_openable("http://127.0.0.1:7341/v0/spec?token=abc&instrument=run&instrument=.."));
}

// The browser's opener is started with the standard descriptors only: a
// descriptor of the app that is not marked close-on-exec, such as a listening
// socket's, never reaches it or the browser it starts.
TEST_CASE("builder app: the browser's opener gets only the standard descriptors") {
    const std::string root = fresh_dir("builder-app-opener");
    const std::string out = root + "/fds.txt";
    int p[2];
    REQUIRE(::pipe(p) == 0);
    // Two descriptors without close-on-exec, as a careless one is, at numbers
    // the child's own (the listing's directory) never take.
    REQUIRE(::dup2(p[0], 57) == 57);
    REQUIRE(::dup2(p[1], 58) == 58);
    std::string why;
    REQUIRE_MESSAGE(caliper_host::spawn_detached({"/bin/sh", "-c", "ls /dev/fd > '" + out + "'"}, &why), why);
    std::string listed;
    for (int i = 0; i < 200 && listed.find('\n') == std::string::npos; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        listed = std::filesystem::exists(out) ? read_text(out) : std::string();
    }
    ::close(57);
    ::close(58);
    ::close(p[0]);
    ::close(p[1]);
    INFO("the child's descriptors: " << listed);
    std::vector<std::string> fds;
    {
        std::istringstream in(listed);
        for (std::string f; in >> f;) fds.push_back(f);
    }
    REQUIRE_FALSE(fds.empty());
    CHECK(std::find(fds.begin(), fds.end(), "57") == fds.end());
    CHECK(std::find(fds.begin(), fds.end(), "58") == fds.end());
    for (const char* std_fd : {"0", "1", "2"}) CHECK(std::find(fds.begin(), fds.end(), std_fd) != fds.end());
}

// Home from a dashboard just made and saved: the landing selects that
// dashboard, not the first instrument, and Projects lists it, the
// workspace's own dashboard, while Instruments keeps the installed instruments.
TEST_CASE("builder app: Home from a new dashboard selects it and lists it under Projects" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_builder_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("builder-app-home");
    const std::string data = root + "/files", shots = root + "/shots";
    for (const auto& d : {shots, root + "/data", root + "/apps"}) std::filesystem::create_directories(d);
    write_parquet(data + "/run42_metrics.parquet",
                  "SELECT range AS step, 0.1 + 2.0 * exp(-range / 50.0) AS loss FROM range(100)");
    const std::string script =
        "wait 60; page landing; landing projects none; "
        "key cmd+k; type Training run; key f3; page spec; "
        "drop " + data + "/run42_metrics.parquet; pane source table:run42_metrics; "
        "click source:table:run42_metrics 0.6 0.5; click primitive:series; pane focus series; "
        "key cmd+s; wait 20; key escape; wait 5; key escape; page landing; wait 20; "
        "landing selected local.training-run; landing projects local.training-run; "
        "leave; wait 5; shot home; quit";
    caliper_builder::acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--workspace", root + "/ws"};
    o.env = {{"CALIPER_DATA_DIR", root + "/data"},
             {"CALIPER_USER_INSTRUMENTS_DIR", root + "/apps"},
             {"CALIPER_WINDOW_SIZE", "1440x900"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", script},
             {"CALIPER_TEST_SPEC_UI_SHOTS", shots},
             {"CALIPER_EXIT_AFTER", "150"}};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 200000;
    lim.max_output_bytes = 16u << 20;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] the landing's selection is local.training-run") != std::string::npos);
    CHECK(run.err.find("[ui] Projects lists local.training-run") != std::string::npos);
    CHECK(run.err.find("[ui] done") != std::string::npos);
    CHECK(std::filesystem::is_regular_file(shots + "/home.png"));
}

// ⌘O Open in browser on a saved dashboard, in an app started without --api:
// the app starts Caliper API on loopback as the side chat does, and gives the
// link of the viewer to the test hook in place of the browser. The link
// carries the viewer token, never the person's session token or the API
// file's: it opens the spec page and reads the dashboard, and nothing else.
// Without the hook, a hidden app refuses, says so, and starts nothing.
TEST_CASE("builder app: Open in browser gives the viewer's link, with the API it starts, to the test hook" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_builder_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const std::string root = fresh_dir("builder-app-browser");
    const std::string shots = root + "/shots", link_file = root + "/link.txt", done = root + "/done";
    for (const auto& d : {shots, root + "/data", root + "/apps"}) std::filesystem::create_directories(d);
    saved_dashboard(root);
    auto spawn = [&](bool hook, const std::string& script) {
        caliper_builder::acp::SpawnOptions o;
        o.argv = {exe, "--hidden", "--workspace", root + "/ws"};
        o.env = {{"CALIPER_DATA_DIR", root + "/data"},
                 {"CALIPER_USER_INSTRUMENTS_DIR", root + "/apps"},
                 {"CALIPER_WINDOW_SIZE", "1440x900"},
                 {"CALIPER_AGENT_OFFLINE", "1"},
                 {"CALIPER_COMMAND_SCRIPT", root + "/build.jsonl"},
                 {"CALIPER_TEST_SPEC_UI", script},
                 {"CALIPER_TEST_SPEC_UI_SHOTS", shots},
                 {"CALIPER_EXIT_AFTER", "150"}};
        if (hook) o.env.push_back({"CALIPER_TEST_OPEN_URL", link_file});
        o.env_remove = {"CALIPER_API"};
        caliper_builder::acp::RunLimits lim;
        lim.timeout_ms = 200000;
        lim.max_output_bytes = 16u << 20;
        return caliper_builder::acp::run_process(o, lim);
    };

    // With the hook: the app runs on while the test opens the link.
    caliper_builder::acp::RunResult run;
    std::atomic<bool> ended{false};
    std::thread app([&] {
        run = spawn(true, "wait 60; page spec; wait 20; click page:browser; wait 20; leave; wait 5; shot browser; "
                          "wait-file " + done + "; quit");
        ended = true;
    });
    std::string url;
    for (int i = 0; i < 1800 && url.empty() && !ended; ++i) {
        std::ifstream in(link_file);
        std::string line;
        if (std::getline(in, line) && line.find("instrument=") != std::string::npos) url = line;
        else std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // The link: the viewer's page on loopback, for this dashboard.
    const std::string base = "http://127.0.0.1:";
    int port = 0;
    std::string token;
    if (!url.empty()) {
        CHECK(url.rfind(base, 0) == 0);
        port = std::atoi(url.c_str() + base.size());
        CHECK(url.find("/v0/spec?") != std::string::npos);
        CHECK(query_value(url, "instrument") == "run42-dashboard");
        token = query_value(url, "token");
        CHECK(token.size() >= 22);
        // It opens the page, and reads the dashboard's spec from the viewer origin.
        const std::string target = url.substr(url.find("/v0/spec"));
        const Answer page = http(port, "GET", target);
        CHECK(page.status == 200);
        CHECK(page.body.find("<html") != std::string::npos);
        const std::string body = R"({"instrument":"run42-dashboard"})";
        const Answer spec = http(port, "POST", "/v0/commands/spec.get",
                                 "Authorization: Bearer " + token + "\r\nContent-Type: application/json\r\nContent-Length: " +
                                     std::to_string(body.size()) + "\r\n",
                                 body);
        CHECK(spec.status == 200);
        CHECK(spec.body.find("\"ok\":true") != std::string::npos);
        CHECK(spec.body.find("curves") != std::string::npos);
        // Nothing else: the API's other endpoints refuse it.
        CHECK(http(port, "GET", "/v0/info?token=" + token).status == 403);
        CHECK(http(port, "GET", "/v0/instruments?token=" + token).status == 403);
        // It is not the API file's token, which outside agents use; the
        // file's spec_url is this same viewer link.
        const json api_file = json::parse(read_text(root + "/data/api.json"), nullptr, false);
        REQUIRE(api_file.is_object());
        CHECK(api_file.value("port", 0) == port);
        CHECK(api_file.value("token", std::string()) != token);
        CHECK(api_file.value("spec_url", std::string()).find("token=" + token) != std::string::npos);
    }
    std::ofstream(done).close();
    app.join();
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    REQUIRE_FALSE(url.empty());
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
    // The API was started for the browser, on loopback, with the line --api
    // prints; the person's session token is on that line and not in the link.
    CHECK(run.err.find("[browser] Caliper API started on 127.0.0.1:" + std::to_string(port)) != std::string::npos);
    const std::string person = token_after(run.out, "with the session token ");
    CHECK(person.size() >= 22);
    CHECK(person != token);
    CHECK(url.find(person) == std::string::npos);
    // The token is never written to standard error or into Activity's lines.
    CHECK(run.err.find(token) == std::string::npos);
    CHECK(std::filesystem::is_regular_file(shots + "/browser.png"));

    // Without the hook a hidden app opens nothing, says so, and starts no API.
    std::filesystem::remove(link_file);
    const auto refused = spawn(false, "wait 60; page spec; wait 20; click page:browser; wait 20; quit");
    INFO("stderr tail: " << refused.err.substr(refused.err.size() > 4000 ? refused.err.size() - 4000 : 0));
    CHECK(refused.status.success());
    CHECK(refused.err.find("[ui] done") != std::string::npos);
    CHECK(refused.err.find("[browser] not opened: the app runs hidden") != std::string::npos);
    CHECK(refused.out.find("[api] Caliper API on") == std::string::npos);
    CHECK_FALSE(std::filesystem::exists(link_file));
}

// ⌘O Open in browser whose link cannot be handed over (here the test hook's
// file cannot be written) gives no link out, so it stops the API that it
// alone started, and its API file goes with it, while the app runs on.
TEST_CASE("builder app: Open in browser stops the API it started when no link is handed over" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-browser-refused");
    if (root.empty()) return;
    saved_dashboard(root);
    const auto run = run_app(exe, root,
                             "wait 60; page spec; wait 20; click page:browser; wait 20; files " + root +
                                 "/data api.json 0; quit",
                             {{"CALIPER_COMMAND_SCRIPT", root + "/build.jsonl"},
                              {"CALIPER_TEST_OPEN_URL", root + "/no-such-folder/link.txt"}});
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[browser] Caliper API started on 127.0.0.1:") != std::string::npos);
    CHECK(run.err.find("[browser] Caliper API stopped on 127.0.0.1:") != std::string::npos);
    CHECK(run.err.find("[ui] done") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(root + "/data/api.json"));
}

// An agent that opens the dashboard the spec page already shows keeps its row
// ACTIVE: the page matches the shown instrument's entry whether it is READY or
// already ACTIVE.
TEST_CASE("builder app: an agent's instrument.open of the dashboard on screen keeps its row ACTIVE" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-active");
    if (root.empty()) return;
    saved_dashboard(root);
    const std::string ready = root + "/ready", done = root + "/done";
    caliper_builder::acp::RunResult run;
    std::atomic<bool> ended{false};
    std::thread app([&] {
        run = run_app(exe, root, "wait 60; page spec; wait 30; touch " + ready + "; wait-file " + done + "; quit",
                      {{"CALIPER_COMMAND_SCRIPT", root + "/build.jsonl"}}, {"--api", "--api-port", "0"});
        ended = true;
    });
    for (int i = 0; i < 1800 && !ended && !std::filesystem::exists(ready); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::string before, after;
    bool opened = false;
    if (std::filesystem::exists(ready)) {
        const json api = json::parse(read_text(root + "/data/api.json"), nullptr, false);
        const int port = api.is_object() ? api.value("port", 0) : 0;
        const std::string auth = "Authorization: Bearer " + (api.is_object() ? api.value("token", std::string()) : "") + "\r\n";
        const auto status_of = [&] {
            const Answer a = http(port, "GET", "/v0/instruments", auth);
            const json j = json::parse(a.body, nullptr, false);
            if (j.is_object() && j.contains("instruments"))
                for (const auto& e : j["instruments"])
                    if (e.value("id", std::string()) == "local.run42-dashboard") return e.value("status", std::string());
            return std::string("missing");
        };
        before = status_of();
        const std::string body = R"({"id":"run42-dashboard"})";
        const Answer open = http(port, "POST", "/v0/commands/instrument.open",
                                 auth + "Content-Type: application/json\r\nContent-Length: " +
                                     std::to_string(body.size()) + "\r\n",
                                 body);
        opened = open.status == 200 && open.body.find("\"ok\":true") != std::string::npos;
        after = status_of();   // a later frame than the one that ran the open
    }
    std::ofstream(done).close();
    app.join();
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(before == "ACTIVE");
    CHECK(opened);
    CHECK(after == "ACTIVE");
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
}

// An agent's new dashboard never ends the person's running instrument: on an
// instrument library's page the spec opens in the session, the page stays, and
// the app says so; Home, then F6 Dashboard on the landing, shows it. The
// instrument is the first one with a library beside the app.
TEST_CASE("builder app: an agent's new dashboard leaves the person's instrument page" * doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-agent-instrument");
    if (root.empty()) return;
    // An instrument with a library beside the app, whichever the build made: the host
    // names none, so its id comes from its manifest.
    std::string instrument;
    std::error_code ec;
    for (const auto& f : std::filesystem::directory_iterator(std::filesystem::path(exe).parent_path() / "instruments", ec)) {
        if (!instrument.empty() || f.path().string().find(".caliper.toml") == std::string::npos) continue;
        std::istringstream in(read_text(f.path().string()));
        for (std::string line; std::getline(in, line) && instrument.empty();) {
            const size_t q = line.find('"');
            if (line.rfind("id", 0) == 0 && q != std::string::npos && line.find('"', q + 1) != std::string::npos)
                instrument = line.substr(q + 1, line.find('"', q + 1) - q - 1);
        }
    }
    if (instrument.empty()) {
        std::printf("caliper_builder_app: the instrument page case is not run: no instrument with a library beside the app\n");
        return;
    }
    const std::string results = root + "/results.jsonl";
    std::ofstream(root + "/agent.jsonl")
        << R"({"name":"spec.new","origin":"agent","args":{"id":"agent-board","title":"Agent board"}})" << "\n";
    const auto run = run_app(exe, root,
                             "wait 30; page instrument; wait-file " + results + "; wait 30; page instrument; leave; wait 5; "
                             "shot instrument; key escape; page landing; wait 10; key f6; page spec; wait 10; shot back; quit",
                             {{"CALIPER_AUTOLAUNCH", instrument},
                              {"CALIPER_COMMAND_SCRIPT", root + "/agent.jsonl"},
                              {"CALIPER_COMMAND_RESULTS", results}});
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(read_text(results).find("\"ok\":true") != std::string::npos);
    CHECK(run.err.find("[spec] an agent opened Agent board; the instrument's page stays") != std::string::npos);
    CHECK(run.err.find("[ui] done") != std::string::npos);
}

// ⌘O opens the saved file, so once a link is out and the dashboard has
// unsaved changes, the page says that the browser shows the saved dashboard.
TEST_CASE("builder app: with unsaved changes the page says the browser shows the saved dashboard" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const std::string root = app_root(exe, "builder-app-browser-note");
    if (root.empty()) return;
    saved_dashboard(root);
    const std::string link = root + "/link.txt";
    const auto run = run_app(exe, root,
                             "wait 60; page spec; wait 20; browser note none; click page:browser; wait-file " + link +
                                 "; browser note none; key cmd+b; wait 10; click prop:spec.title; key cmd+a; "
                                 "type Run 43; key enter; browser note the browser shows the saved dashboard; "
                                 "leave; wait 5; shot note; quit",
                             {{"CALIPER_COMMAND_SCRIPT", root + "/build.jsonl"}, {"CALIPER_TEST_OPEN_URL", link}});
    INFO("stderr tail: " << run.err.substr(run.err.size() > 4000 ? run.err.size() - 4000 : 0));
    REQUIRE(run.error.ok());
    if (run.err.find("[renderer] metal") == std::string::npos) {
        std::printf("caliper_builder_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] the browser note is \"the browser shows the saved dashboard\"") != std::string::npos);
    CHECK(run.err.find("[ui] done") != std::string::npos);
}

#endif
