// The side chat in the hidden app (the hidden-app pattern of
// docs/explanations/testing.md), with the stub agents: no real agent and no
// model runs, and CALIPER_AGENT_OFFLINE=1 is set for the app. Two cases:
//
//   - The spec page: a dashboard built by a command script, the chat opened
//     with ⌘L Chat's toggle, a turn in which the stub agent adds a panel
//     through caliper-mcp, held for review and approved through the
//     approval cards' test-only decisions file, Ask in chat from ⌘K with its
//     draft, and the agent ending. The app was started without --api, so the
//     chat starts Caliper API, on 127.0.0.1 whatever its settings file says,
//     and stops it when the agent ends.
//   - The acceptance case "plot attention entropy per layer while it trains":
//     the app attached to caliper_attention_producer's stream and started
//     with --api, and the person's message is that sentence. The agent is a
//     script (caliper_chat_stub_agent): it lists the live sources and builds
//     the spec through caliper-mcp, each edit approved on the cards, and
//     opens it; the panels then follow the stream. The per-head entropy is
//     computed by the producer; the spec computes the entropy of each layer's
//     head-mean attention. The chat leaves the API alone, since the person
//     started it.
//
// They need the caliper executable (CALIPER_TEST_CALIPER_EXE) and a Metal
// device, and the second the producer (CALIPER_TEST_PRODUCER_EXE) and the
// test applets (CALIPER_TEST_APPLETS); without them they say they are skipped.
#include <stb_image.h>

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/commands/journal.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;
namespace acp = caliper_builder::acp;

#ifndef _WIN32

namespace {

fs::path fresh(const std::string& name) {
    const fs::path dir = fs::path(CALIPER_SPEC_TEST_TMP) / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return fs::canonical(dir);
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct Png {
    int w = 0, h = 0;
    std::vector<unsigned char> px;
};

Png load(const std::string& path) {
    Png img;
    int ch = 0;
    unsigned char* px = stbi_load(path.c_str(), &img.w, &img.h, &ch, 4);
    if (!px) return img;
    img.px.assign(px, px + (size_t)img.w * img.h * 4);
    stbi_image_free(px);
    return img;
}

// Pixels that differ between two images in the columns [0, x1).
size_t differ_left(const Png& a, const Png& b, double share) {
    if (a.w != b.w || a.h != b.h) return 0;
    const int x1 = (int)(a.w * share);
    size_t n = 0;
    for (int y = 0; y < a.h; ++y)
        for (int x = 0; x < x1; ++x) {
            const size_t i = ((size_t)y * a.w + x) * 4;
            if (a.px[i] != b.px[i] || a.px[i + 1] != b.px[i + 1] || a.px[i + 2] != b.px[i + 2]) ++n;
        }
    return n;
}

size_t count(const std::string& text, const std::string& what) {
    size_t n = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
    return n;
}

// The journal's commands from the origin agent, by name.
std::map<std::string, int> agent_commands(const fs::path& ws) {
    std::map<std::string, int> out;
    const auto journal = caliper_builder::read_journal((ws / "journal" / "commands.jsonl").string());
    for (const auto& e : journal.events)
        if (e.origin == caliper_builder::CommandOrigin::agent) ++out[e.name];
    return out;
}

bool metal(const std::string& err) { return err.find("[renderer] metal") != std::string::npos; }

}  // namespace

TEST_CASE("chat app: the side chat on the hidden app's spec page, with the cards, Ask in chat and the API it starts" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    if (!exe || !*exe) {
        std::printf("caliper_side_chat_app skipped: needs CALIPER_TEST_CALIPER_EXE\n");
        return;
    }
    const char* offline = std::getenv("CALIPER_AGENT_OFFLINE");
    REQUIRE_MESSAGE((offline && std::string(offline) == "1"), "run this test through ctest, which sets the switch");
    const fs::path root = fresh("chat-app");
    const fs::path ws = root / "ws", data = root / "data", gates = root / "gates", shots = root / "shots";
    for (const fs::path& d : {ws / "data", data, gates, shots, root / "apps"}) fs::create_directories(d);
    {
        std::ofstream csv(ws / "data" / "run42_metrics.csv");
        csv << "step,loss,accuracy,lr\n";
        for (int i = 0; i < 200; ++i)
            csv << i << "," << 0.1 + 2.0 * std::exp(-i / 50.0) << "," << 0.1 + 0.8 * (1 - std::exp(-i / 40.0)) << ","
                << 3e-4 * std::min(1.0, (i + 1) / 20.0) << "\n";
    }
    const json build = json::array(
        {{{"name", "data.import"}, {"args", {{"paths", {"data/run42_metrics.csv"}}, {"run", true}}}},
         {{"name", "spec.new"}, {"args", {{"id", "run42-dashboard"}, {"title", "Run 42"}}}},
         {{"name", "source.add"}, {"args", {{"id", "metrics"}, {"catalog", "table:run42_metrics"}}}},
         {{"name", "panel.create"},
          {"args", {{"id", "curves"}, {"primitive", "series"}, {"title", "Loss and accuracy"},
                    {"bind", {{"x", "metrics.step"}, {"y", {"metrics.loss", "metrics.accuracy"}}}}}}},
         {{"name", "panel.create"},
          {"args", {{"id", "rows"}, {"primitive", "table"}, {"bind", {{"rows", "metrics"}}},
                    {"place", {{"target", "curves"}, {"where", "below"}}}}}},
         {{"name", "applet.save"}, {"args", {{"overwrite", true}}}},
         {{"name", "applet.open"}, {"args", {{"id", "run42-dashboard"}}}}});
    {
        std::ofstream f(root / "build.jsonl");
        for (const json& l : build) f << l.dump() << "\n";
    }
    // The person's settings, with keys the app does not know: none of them
    // may move the API off loopback or onto another port.
    {
        json s = {{"agent", "custom"},
                  {"custom_name", "Stub agent"},
                  {"custom_command", "'" + std::string(CALIPER_CHAT_STUB_AGENT) + "' --gate-dir '" + gates.string() + "'"},
                  {"open", false},
                  {"api_address", "0.0.0.0"},
                  {"api_port", 80}};
        std::ofstream(data / "side-chat.json") << s.dump();
    }
    const fs::path decisions = root / "decisions.txt";
    std::ofstream(decisions).close();
    const std::string ui =
        "wait 60; focus rows; wait 3; leave; key escape; wait 10; chat open; wait 10; chat focus; wait 5; shot t-1; "
        "type Add a panel of the learning rate; wait 3; key enter; chat item tool catalog.describe in_progress; "
        "wait 5; shot t-2; touch " + (gates / "turn").string() + "; chat item approval panel.create waiting; "
        "wait 40; shot t-3; append " + decisions.string() + " approve panel.create agent; chat item stop - DONE; "
        "wait 10; shot t-4; focus curves; wait 3; leave; key cmd+k; until unavailable; wait 3; "
        "type Build a loader for the validation logs; wait 3; key enter; chat draft; wait 5; shot t-5; "
        "chat focus; wait 3; key enter; wait 30; chat until ready; chat focus; wait 3; "
        "type copy-api; wait 3; key enter; wait 30; chat until ready; chat focus; wait 3; "
        "type Compare the first and the last layer; wait 3; key enter; chat until exited; wait 10; quit";
    acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--workspace", ws.string()};
    o.env = {{"CALIPER_DATA_DIR", data.string()},
             {"CALIPER_USER_APPLETS_DIR", (root / "apps").string()},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_COMMAND_SCRIPT", (root / "build.jsonl").string()},
             {"CALIPER_TEST_APPROVALS", decisions.string()},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", ui},
             {"CALIPER_TEST_SPEC_UI_SHOTS", shots.string()},
             {"CALIPER_EXIT_AFTER", "150"}};
    o.env_remove = {"CALIPER_API", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"};
    acp::RunLimits lim;
    lim.timeout_ms = 200000;
    const auto run = acp::run_process(o, lim);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 6000 ? run.err.size() - 6000 : 0));
    REQUIRE(run.error.ok());
    if (!metal(run.err)) {
        std::printf("caliper_side_chat_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
    CHECK(run.err.find("[ui] stopped") == std::string::npos);
    for (const char* n : {"t-1", "t-2", "t-3", "t-4", "t-5"}) {
        const Png p = load((shots / (std::string(n) + ".png")).string());
        CHECK_MESSAGE(((p.w == 960 && p.h == 600) || (p.w == 1920 && p.h == 1200)), n);
    }
    // The agent's panel.create reached the command layer from the origin
    // agent, after the person's approval on the cards.
    CHECK(agent_commands(ws)["panel.create"] == 1);
    CHECK(run.err.find("panel.create (agent) approved") != std::string::npos);
    // Ask in chat put a draft in the chat's input.
    CHECK(run.err.find("chat draft of") != std::string::npos);
    // The chat started Caliper API on loopback, whatever the settings say,
    // and stopped it when its agent ended.
    const std::string started = "[chat] Caliper API started on 127.0.0.1:";
    REQUIRE(run.err.find(started) != std::string::npos);
    const std::string port = run.err.substr(run.err.find(started) + started.size(), 5);
    CHECK(port.rfind("80\n", 0) != 0);
    CHECK(run.out.find("[api] Caliper API on http://127.0.0.1:") != std::string::npos);   // the line --api prints
    CHECK(run.out.find("0.0.0.0") == std::string::npos);
    CHECK(run.err.find("0.0.0.0") == std::string::npos);
    CHECK(run.err.find("[chat] Caliper API stopped on 127.0.0.1:") != std::string::npos);
    CHECK_FALSE(fs::exists(data / "api.json"));
    // The agent reached Caliper with a token of its own, origin agent, in an
    // API file of its own, which went when its session ended; the person's
    // token, printed on the start line, is in none of what the agent was given
    // (its API file, the app's api.json, its environment), and api.json holds
    // a token for outside agents, not the person's and not the agent's.
    {
        const std::string mark = "with the session token ";
        const size_t at = run.out.find(mark);
        REQUIRE(at != std::string::npos);
        size_t e = at + mark.size();
        while (e < run.out.size() && (std::isalnum((unsigned char)run.out[e]) || run.out[e] == '_' || run.out[e] == '-')) ++e;
        const std::string person = run.out.substr(at + mark.size(), e - at - mark.size());
        REQUIRE(person.size() >= 22);
        const json agent_file = json::parse(read_file(gates / "api-seen.json"), nullptr, false);
        const json api_file = json::parse(read_file(gates / "api-json-seen.json"), nullptr, false);
        REQUIRE(agent_file.is_object());
        REQUIRE(api_file.is_object());
        const std::string agent_token = agent_file["token"];
        CHECK(agent_token.size() >= 22);
        CHECK(agent_token != person);
        CHECK(agent_token != api_file["token"].get<std::string>());
        CHECK(api_file["token"].get<std::string>() != person);
        CHECK(read_file(gates / "env-seen.txt").find(person) == std::string::npos);
        CHECK(read_file(gates / "env-seen.txt").find(agent_token) == std::string::npos);
        const std::string agent_path = read_file(gates / "api-seen-path.txt");
        CHECK(agent_path.find("api-agent-") != std::string::npos);
        CHECK_FALSE(fs::exists(agent_path));
        // The token went into no argument of caliper-mcp: only the file's path did.
        CHECK(run.err.find(agent_token) == std::string::npos);
        CHECK(run.out.find(agent_token) == std::string::npos);
    }
    // The settings were saved with the pane open, and only the keys the app writes.
    const std::string saved = read_file(data / "side-chat.json");
    CHECK(saved.find("\"open\": true") != std::string::npos);
    CHECK(saved.find("api_address") == std::string::npos);
    CHECK(saved.find(CALIPER_CHAT_STUB_AGENT) != std::string::npos);
}

TEST_CASE("chat app: plot attention entropy per layer while it trains, from the in-app chat (the agent is a script)" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* producer_exe = std::getenv("CALIPER_TEST_PRODUCER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !producer_exe || !*producer_exe || !applets || !*applets) {
        std::printf("caliper_side_chat_app skipped: needs CALIPER_TEST_CALIPER_EXE, CALIPER_TEST_PRODUCER_EXE and "
                    "CALIPER_TEST_APPLETS\n");
        return;
    }
    const fs::path root = fresh("chat-live");
    const fs::path ws = root / "ws", data = root / "data", shots = root / "shots";
    for (const fs::path& d : {ws, data, shots, root / "apps"}) fs::create_directories(d);
    {
        json s = {{"agent", "custom"},
                  {"custom_name", "Stub agent"},
                  {"custom_command", "'" + std::string(CALIPER_CHAT_STUB_AGENT) + "'"},
                  {"open", false}};
        std::ofstream(data / "side-chat.json") << s.dump();
    }
    // Each spec edit the agent sends waits for review; the person approves each.
    const fs::path decisions = root / "decisions.txt";
    {
        std::ofstream d(decisions);
        for (int i = 0; i < 10; ++i) d << "approve source.add agent\n";
        for (int i = 0; i < 3; ++i) d << "approve panel.create agent\n";
        d << "approve layout.set agent\napprove spec.new agent\napprove applet.save agent\n";
    }

    // The training script's stand-in: 6 layers of 8 heads over 64 tokens.
    acp::SpawnOptions po;
    po.argv = {producer_exe, "--layers", "6", "--heads", "8", "--tokens", "64", "--hz", "8"};
    caliper_builder::Error e;
    std::unique_ptr<acp::Process> producer = acp::Process::spawn(po, &e);
    REQUIRE_MESSAGE(producer, e.message);
    std::string out, link;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (link.empty() && std::chrono::steady_clock::now() < until) {
        char b[4096];
        std::size_t got = 0;
        for (acp::Process* p : {producer.get()}) {
            if (p->read_stdout(b, sizeof b, &got, 100) == acp::IoStatus::ok) out.append(b, got);
            if (p->read_stderr(b, sizeof b, &got, 10) == acp::IoStatus::ok) out.append(b, got);
        }
        const size_t at = out.find("[producer] hub ");
        if (at != std::string::npos && out.find('\n', at) != std::string::npos) {
            const size_t s = at + std::string("[producer] hub ").size();
            link = out.substr(s, out.find('\n', at) - s);
        }
    }
    REQUIRE_MESSAGE(!link.empty(), out);

    const std::string ui =
        "wait 120; chat open; wait 10; chat focus; wait 5; type Plot attention entropy per layer while it trains; "
        "wait 3; key enter; chat item tool catalog.list completed; wait 5; shot live-1; chat item stop - DONE; "
        "wait 60; shot live-2a; wait 90; shot live-2b; chat focus; wait 3; "
        "type Compare the first and the last layer; wait 3; key enter; chat until exited; wait 30; quit";
    acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--applets", applets, "--url", link, "--api", "--api-port", "0", "--workspace", ws.string()};
    o.env = {{"CALIPER_DATA_DIR", data.string()},
             {"CALIPER_USER_APPLETS_DIR", (root / "apps").string()},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_TEST_APPROVALS", decisions.string()},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", ui},
             {"CALIPER_TEST_SPEC_UI_SHOTS", shots.string()},
             {"CALIPER_EXIT_AFTER", "150"}};
    o.env_remove = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"};
    acp::RunLimits lim;
    lim.timeout_ms = 200000;
    const auto run = acp::run_process(o, lim);
    producer->terminate(2000);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 6000 ? run.err.size() - 6000 : 0));
    REQUIRE(run.error.ok());
    if (!metal(run.err)) {
        std::printf("caliper_side_chat_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
    CHECK(run.err.find("[ui] stopped") == std::string::npos);
    // The script built the spec over the live sources through the command
    // layer, every edit approved on the cards, and opened it.
    std::map<std::string, int> cmds = agent_commands(ws);
    CHECK(cmds["spec.new"] == 1);
    CHECK(cmds["source.add"] == 9);   // the entropy, the attention, the layers' entropy and one per layer
    CHECK(cmds["panel.create"] == 3);
    CHECK(cmds["layout.set"] == 1);
    CHECK(count(run.err, "(agent) approved") >= 13);
    CHECK(run.err.find("[ui] shot live-2b") != std::string::npos);
    CHECK(run.err.find("chat ready") != std::string::npos);
    // While it trains: the panels change between two frames a moment apart.
    const Png a = load((shots / "live-2a.png").string()), b = load((shots / "live-2b.png").string());
    REQUIRE(a.w > 0);
    CHECK(differ_left(a, b, 0.6) > 500);
    // The person started the API with --api: the chat neither started nor stopped it.
    CHECK(run.err.find("[chat] Caliper API started") == std::string::npos);
    CHECK(run.err.find("[chat] Caliper API stopped") == std::string::npos);
}

TEST_CASE("chat app: the Shared GPU note while a live run is attached and the agent is marked as local" *
          doctest::test_suite("app")) {
    const char* exe = std::getenv("CALIPER_TEST_CALIPER_EXE");
    const char* producer_exe = std::getenv("CALIPER_TEST_PRODUCER_EXE");
    const char* applets = std::getenv("CALIPER_TEST_APPLETS");
    if (!exe || !*exe || !producer_exe || !*producer_exe || !applets || !*applets) {
        std::printf("caliper_side_chat_app skipped: needs CALIPER_TEST_CALIPER_EXE, CALIPER_TEST_PRODUCER_EXE and "
                    "CALIPER_TEST_APPLETS\n");
        return;
    }
    const fs::path root = fresh("chat-gpu-note");
    const fs::path ws = root / "ws", data = root / "data", shots = root / "shots";
    for (const fs::path& d : {ws, data, shots, root / "apps"}) fs::create_directories(d);
    // The stub as the person's agent, not marked yet; it never starts, since
    // no message is sent.
    {
        json s = {{"agent", "custom"},
                  {"custom_name", "Stub agent"},
                  {"custom_command", "'" + std::string(CALIPER_CHAT_STUB_AGENT) + "'"},
                  {"open", false}};
        std::ofstream(data / "side-chat.json") << s.dump();
    }
    // The training script's stand-in.
    acp::SpawnOptions po;
    po.argv = {producer_exe, "--layers", "2", "--heads", "4", "--tokens", "16", "--hz", "8"};
    caliper_builder::Error e;
    std::unique_ptr<acp::Process> producer = acp::Process::spawn(po, &e);
    REQUIRE_MESSAGE(producer, e.message);
    std::string out, link;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (link.empty() && std::chrono::steady_clock::now() < until) {
        char b[4096];
        std::size_t got = 0;
        if (producer->read_stdout(b, sizeof b, &got, 100) == acp::IoStatus::ok) out.append(b, got);
        if (producer->read_stderr(b, sizeof b, &got, 10) == acp::IoStatus::ok) out.append(b, got);
        const size_t at = out.find("[producer] hub ");
        if (at != std::string::npos && out.find('\n', at) != std::string::npos) {
            const size_t s = at + std::string("[producer] hub ").size();
            link = out.substr(s, out.find('\n', at) - s);
        }
    }
    REQUIRE_MESSAGE(!link.empty(), out);

    // Attached, the note waits for the mark; marked, it shows; detached (esc
    // Home on the live page), it goes.
    const std::string ui =
        "page live; chat open; chat until idle; chat note none; chat local on; chat note Shared GPU; wait 5; shot gpu; "
        "key escape; page landing; chat note none; quit";
    acp::SpawnOptions o;
    o.argv = {exe, "--hidden", "--applets", applets, "--url", link, "--workspace", ws.string()};
    o.env = {{"CALIPER_DATA_DIR", data.string()},
             {"CALIPER_USER_APPLETS_DIR", (root / "apps").string()},
             {"CALIPER_WINDOW_SIZE", "960x600"},
             {"CALIPER_AGENT_OFFLINE", "1"},
             {"CALIPER_TEST_SPEC_UI", ui},
             {"CALIPER_TEST_SPEC_UI_SHOTS", shots.string()},
             {"CALIPER_EXIT_AFTER", "100"}};
    o.env_remove = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"};
    acp::RunLimits lim;
    lim.timeout_ms = 150000;
    const auto run = acp::run_process(o, lim);
    producer->terminate(2000);
    INFO("stderr tail: " << run.err.substr(run.err.size() > 6000 ? run.err.size() - 6000 : 0));
    REQUIRE(run.error.ok());
    if (!metal(run.err)) {
        std::printf("caliper_side_chat_app skipped: no Metal device\n");
        return;
    }
    CHECK(run.status.success());
    CHECK(run.err.find("[ui] done") != std::string::npos);
    CHECK(run.err.find("[ui] stopped") == std::string::npos);
    const size_t marked = run.err.find("[ui] chat local on");
    const size_t shown = run.err.find("[ui] the chat's note is Shared GPU");
    REQUIRE(marked != std::string::npos);
    REQUIRE(shown != std::string::npos);
    CHECK(marked < shown);
    CHECK(run.err.find("[ui] the chat's note is none", shown) != std::string::npos);
    const Png p = load((shots / "gpu.png").string());
    CHECK(((p.w == 960 && p.h == 600) || (p.w == 1920 && p.h == 1200)));
    // The mark is kept in the chat's settings, for the chosen agent.
    const json saved = json::parse(read_file(data / "side-chat.json"), nullptr, false);
    REQUIRE(saved.is_object());
    CHECK(saved.value("local_model_agents", json::array()) == json::array({"custom"}));
}

#endif
