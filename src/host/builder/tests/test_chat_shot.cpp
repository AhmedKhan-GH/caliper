// caliper-chat-shot as a process, on an offscreen canvas: the capture
// scenario with the stub agent driving Caliper through caliper-mcp. The JSON
// it prints says what the conversation held, and the journal says that the
// dashboard gained the panel through the command layer from the origin
// agent; the images say that the panel appeared in the dashboard's part of
// the window and that the chat drew in its own.
#include <stb_image.h>

#include <caliper_builder/acp/subprocess.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

struct Png {
    int w = 0, h = 0;
    std::vector<unsigned char> px;
};

Png load(const std::string& path) {
    Png img;
    int ch = 0;
    unsigned char* px = stbi_load(path.c_str(), &img.w, &img.h, &ch, 4);
    REQUIRE_MESSAGE(px, path);
    img.px.assign(px, px + (size_t)img.w * img.h * 4);
    stbi_image_free(px);
    return img;
}

// Pixels that differ inside the columns [x0, x1).
size_t differ(const Png& a, const Png& b, int x0, int x1) {
    REQUIRE(a.w == b.w);
    REQUIRE(a.h == b.h);
    size_t n = 0;
    for (int y = 0; y < a.h; ++y)
        for (int x = x0; x < x1; ++x) {
            const size_t i = ((size_t)y * a.w + x) * 4;
            if (a.px[i] != b.px[i] || a.px[i + 1] != b.px[i + 1] || a.px[i + 2] != b.px[i + 2]) ++n;
        }
    return n;
}

struct Run {
    bool skipped = false;
    json summary;
    std::string out_dir;
};

Run run_shot(const std::string& name, const std::string& review) {
    Run r;
    const fs::path root = fs::path(CALIPER_SPEC_TEST_TMP) / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "shots");
    r.out_dir = (root / "shots").string();
    caliper_builder::acp::SpawnOptions o;
    o.argv = {CALIPER_CHAT_SHOT_EXE, "--workspace", (root / "ws").string(), "--agent", CALIPER_CHAT_STUB_AGENT,
              "--out-dir", r.out_dir, "--size", "960x600", "--frames", "3", "--review", review, "--mcp", CALIPER_MCP_EXE};
    o.env = {{"CALIPER_AGENT_OFFLINE", "1"}};
    caliper_builder::acp::RunLimits lim;
    lim.timeout_ms = 240000;
    const auto run = caliper_builder::acp::run_process(o, lim);
    INFO("stderr: " << run.err);
    REQUIRE(run.error.ok());
    if (!run.status.success() && run.err.find("cannot draw offscreen") != std::string::npos) {
        r.skipped = true;
        return r;
    }
    REQUIRE(run.status.success());
    const size_t nl = run.out.rfind('\n', run.out.size() - 2);
    r.summary = json::parse(nl == std::string::npos ? run.out : run.out.substr(nl + 1));
    return r;
}

const json* item(const json& items, const std::string& kind, const std::string& command = "") {
    for (const json& it : items)
        if (it["kind"] == kind && (command.empty() || it.value("command", "") == command)) return &it;
    return nullptr;
}

}  // namespace

TEST_CASE("caliper-chat-shot: the side chat's scenario, the agent adding a panel through the command layer" *
          doctest::test_suite("gpu")) {
    const Run r = run_shot("chat-shot", "off");
    if (r.skipped) {
        MESSAGE("skipped: no offscreen canvas");
        return;
    }
    const json& s = r.summary;
    INFO(s.dump(2));
    std::vector<std::string> names, states;
    for (const auto& f : s["files"]) {
        names.push_back(f["name"]);
        states.push_back(f["state"]);
    }
    CHECK(names == std::vector<std::string>{"1-empty", "2-turn", "3-permission", "4-done", "5-exited", "6-no-agent"});
    CHECK(states == std::vector<std::string>{"idle", "running", "running", "ready", "exited", "no_agent"});

    // The panel reached the open spec through panel.create from the origin agent.
    CHECK(s["journal_agent_panel_create"] == true);
    CHECK(s["panels"] == json::array({"curves", "attention", "rows", "lr"}));

    // The conversation.
    const json& items = s["items"];
    REQUIRE(item(items, "person"));
    CHECK((*item(items, "person"))["text"] == "Add a panel of the learning rate");
    REQUIRE(item(items, "tool", "spec.get"));
    CHECK((*item(items, "tool", "spec.get"))["status"] == "completed");
    REQUIRE(item(items, "tool", "catalog.describe"));
    REQUIRE(item(items, "tool", "panel.create"));
    CHECK((*item(items, "tool", "panel.create"))["status"] == "completed");
    REQUIRE(item(items, "permission", "panel.create"));
    CHECK((*item(items, "permission", "panel.create"))["answer"] == "Allow");
    REQUIRE(item(items, "stop"));
    CHECK((*item(items, "stop"))["text"] == "DONE");
    REQUIRE(item(items, "notice"));
    CHECK((*item(items, "notice"))["text"] == "EXITED");
    CHECK((*item(items, "notice"))["detail"] == "exit code 1");
    CHECK(s["header_actions"] == json::array({"Restart"}));
    CHECK(s["no_agent_state"] == "no_agent");

    // The permission card had the dry run's diff, and no allow_always by default.
    CHECK(s["permission"]["command"] == "panel.create");
    CHECK(s["permission"]["diff_path"] == "run42-dashboard.spec.toml");
    CHECK(s["permission"]["diff"].get<std::string>().find("+id = \"lr\"") != std::string::npos);
    CHECK(s["permission"]["options"] == json::array({"allow_once", "reject_once"}));

    // The images: the panel appeared in the dashboard's part between the
    // permission and the end of the turn, and the chat's part changed at each step.
    const Png turn = load(r.out_dir + "/chat-2-turn-960x600.png");
    const Png permission = load(r.out_dir + "/chat-3-permission-960x600.png");
    const Png done = load(r.out_dir + "/chat-4-done-960x600.png");
    const Png empty = load(r.out_dir + "/chat-1-empty-960x600.png");
    CHECK(turn.w == 960);
    CHECK(turn.h == 600);
    const int column = 960 - 300;   // the column's minimum at text scale 1 is wider than 28 % of 960
    CHECK(differ(turn, permission, 0, column) == 0);
    CHECK(differ(permission, done, 0, column) > 2000);
    CHECK(differ(empty, turn, column, 960) > 2000);
    CHECK(differ(turn, permission, column, 960) > 2000);
}

TEST_CASE("caliper-chat-shot: with review on, the agent's edit waits for the approval mirrored in the chat" *
          doctest::test_suite("gpu")) {
    const Run r = run_shot("chat-shot-review", "on");
    if (r.skipped) {
        MESSAGE("skipped: no offscreen canvas");
        return;
    }
    const json& s = r.summary;
    INFO(s.dump(2));
    CHECK(s["approval"]["command"] == "panel.create");
    CHECK(s["journal_agent_panel_create"] == true);
    CHECK(s["panels"] == json::array({"curves", "attention", "rows", "lr"}));
    // No permission card: the agent did not ask a second time for what Caliper reviews.
    CHECK(item(s["items"], "permission") == nullptr);
    REQUIRE(item(s["items"], "approval", "panel.create"));
    CHECK((*item(s["items"], "approval", "panel.create"))["answer"] == "decided");
}

TEST_CASE("caliper-chat-shot: the command line's refusals" * doctest::test_suite("gpu")) {
    caliper_builder::acp::SpawnOptions o;
    o.argv = {CALIPER_CHAT_SHOT_EXE, "--workspace", "x"};
    const auto run = caliper_builder::acp::run_process(o, {});
    REQUIRE(run.error.ok());
    CHECK(run.status.code == 2);
    CHECK(run.err.find("usage") != std::string::npos);
}
