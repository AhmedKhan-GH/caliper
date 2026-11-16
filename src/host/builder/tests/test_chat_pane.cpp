// The side chat in the app (chat_pane.h), headless: its settings file, the
// toggle and its key, the column it claims after the page's strips so the
// canvas keeps the rest, Ask in chat opening it with the handoff draft, a
// command approval of the agent's turn as one row, the line it writes when
// the app starts Caliper API for it, and the hook that lets the app stop
// that API when the agent ends. A real AppSession runs on the test's thread
// as the app's frame loop does; the agents are the stub agents.
#include "builder/app_session.h"
#include "builder/chat_pane.h"
#include "spec_runtime/spec_runtime.h"
#include "ui/sternbild_fonts.h"
#include "ui/sternbild_theme.h"
#include "ui/work_area.h"

#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <thread>

using namespace caliper_host;
namespace fs = std::filesystem;
using nlohmann::json;

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

// A headless ImGui context with the house faces, as the app has.
struct Ui {
    ImGuiContext* ctx = nullptr;
    static constexpr float kW = 1440, kH = 900, kPageBar = 28, kStatusBar = 22, kStrip = 120;
    Ui() {
        ctx = ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.DisplaySize = ImVec2(kW, kH);
        io.DeltaTime = 1.0f / 60.0f;
        sternbild::FontPaths fonts = sternbild::default_font_paths();
        fonts.display_font.clear();
        sternbild::load_fonts(fonts);
        sternbild::apply_style();
    }
    ~Ui() { ImGui::DestroyContext(ctx); }
    static void bar(const char* name, ImGuiDir dir, float h) {
        ImGui::BeginViewportSideBar(name, ImGui::GetMainViewport(), dir, h, ImGuiWindowFlags_NoSavedSettings);
        ImGui::End();
    }
    // One frame of a page: its bars and strips, then `inside`, which draws
    // the pane's column and reads the work area left.
    void frame(const std::function<void()>& inside) {
        ImGui::NewFrame();
        bar("##page_bar", ImGuiDir_Up, kPageBar);
        bar("##status_bar", ImGuiDir_Down, kStatusBar);
        bar("##approvals", ImGuiDir_Down, kStrip);
        inside();
        ImGui::Render();
        for (ImTextureData* tex : ImGui::GetPlatformIO().Textures) {
            if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates) {
                tex->SetTexID((ImTextureID)(intptr_t)1);
                tex->SetStatus(ImTextureStatus_OK);
            } else if (tex->Status == ImTextureStatus_WantDestroy) {
                tex->SetTexID(ImTextureID_Invalid);
                tex->SetStatus(ImTextureStatus_Destroyed);
            }
        }
    }
};

// The app's session on this thread, as the app's frame loop runs it.
struct App {
    AppSession app;
    explicit App(const fs::path& ws) {
        AppSession::Options o;
        o.workspace = ws.string();
        o.chosen = true;
        o.from = "--workspace";
        AppSession::Hooks h;
        h.err = &err;
        app.start(o, h);
    }
    ~App() { app.close(); }
    bool run(ChatPane& pane, const std::function<bool()>& until, int ms = 20000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (std::chrono::steady_clock::now() < end) {
            app.frame();
            pane.frame(app);
            if (until()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        return until();
    }
    std::ostringstream err;
};

ChatPaneSettings stub_settings(const std::string& args = "") {
    ChatPaneSettings s;
    s.chat.agent = "custom";
    s.chat.custom_name = "Stub agent";
    s.chat.custom_command = std::string("'") + CALIPER_ACP_STUB_AGENT + "'" + (args.empty() ? "" : " " + args);
    return s;
}

caliper_builder::CommandResult invoke(caliper_builder::Session& s, const std::string& name, json args,
                                      caliper_builder::CommandOrigin origin = caliper_builder::CommandOrigin::ui) {
    caliper_builder::CommandInvocation inv;
    inv.name = name;
    inv.args = std::move(args);
    inv.origin = origin;
    return s.invoke(std::move(inv));
}

}  // namespace

TEST_CASE("chat pane: the settings file holds the agent and the custom command, never a key or an address") {
    const fs::path dir = fresh("chat-pane-settings");
    const std::string path = (dir / "side-chat.json").string();
    ChatPaneSettings s;
    CHECK(read_chat_settings(path, &s));   // missing: the defaults
    CHECK(s.chat.agent.empty());
    CHECK(s.chat.custom_name == "Custom agent");
    CHECK_FALSE(s.open);

    s = stub_settings("--flag");
    s.chat.allow_always = true;
    s.open = true;
    REQUIRE(write_chat_settings(path, s));
    ChatPaneSettings r;
    REQUIRE(read_chat_settings(path, &r));
    CHECK(r.chat.agent == "custom");
    CHECK(r.chat.custom_name == "Stub agent");
    CHECK(r.chat.custom_command == s.chat.custom_command);
    CHECK(r.chat.allow_always);
    CHECK(r.open);

    // Keys it does not know are not read, and are gone once it writes: no
    // setting chooses the API's address, and none holds a key.
    std::ofstream(path) << R"({"agent": "custom", "api_address": "0.0.0.0", "api_port": 80, "api_key": "sk-test"})";
    REQUIRE(read_chat_settings(path, &r));
    CHECK(r.chat.agent == "custom");
    REQUIRE(write_chat_settings(path, r));
    const std::string text = read_file(path);
    CHECK(text.find("0.0.0.0") == std::string::npos);
    CHECK(text.find("api_port") == std::string::npos);
    CHECK(text.find("sk-test") == std::string::npos);

    std::ofstream(path) << "not json";
    std::string why;
    CHECK_FALSE(read_chat_settings(path, &r, &why));
    CHECK_FALSE(why.empty());
    CHECK(r.chat.agent.empty());
}

TEST_CASE("chat pane: the toggle and ⌘L, and the column that leaves the canvas the rest of the page") {
    const fs::path dir = fresh("chat-pane-toggle");
    fs::create_directories(dir / "empty");
    const std::string path = (dir / "side-chat.json").string();
    REQUIRE(write_chat_settings(path, stub_settings()));
    Ui ui;
    App a(dir / "ws");
    ChatPane pane;
    ChatPane::Options o;
    o.settings_path = path;
    o.path = (dir / "empty").string();
    pane.start(o, {});
    CHECK_FALSE(pane.open());
    CHECK(pane.state() == "none");   // made when it first opens

    pane.toggle();
    CHECK(pane.open());
    ChatPaneSettings saved;
    REQUIRE(read_chat_settings(path, &saved));
    CHECK(saved.open);   // the toggle's state is kept
    REQUIRE(a.run(pane, [&] { return pane.state() != "none"; }));
    CHECK(pane.state() == "idle");

    // Open: the column claims the right edge after the strips; the page's
    // work area is what layout_chat_page() gives the canvas.
    const ChatLayoutMetrics m = ChatLayoutMetrics::for_text_scale(1.0f);
    const ChatColumnLayout want = layout_chat_page({0, 0, Ui::kW, Ui::kH}, Ui::kPageBar, Ui::kStatusBar + Ui::kStrip, true, false, m);
    ImRect left;
    ui.frame([&] {
        pane.draw_column();
        left = work_area();
    });
    CHECK(left.Min.x == doctest::Approx(want.canvas.x));
    CHECK(left.Max.x == doctest::Approx(want.canvas.right()));
    CHECK(left.Min.y == doctest::Approx(want.canvas.y));
    CHECK(left.Max.y == doctest::Approx(want.canvas.bottom()));
    ImGuiWindow* column = ImGui::FindWindowByName("##caliper_side_chat");
    REQUIRE(column != nullptr);
    CHECK(column->Pos.x == doctest::Approx(want.chat.x));
    CHECK(column->Size.x == doctest::Approx(want.chat.w));
    CHECK(column->Pos.y == doctest::Approx(Ui::kPageBar));
    CHECK(column->Pos.y + column->Size.y == doctest::Approx(Ui::kH - Ui::kStatusBar - Ui::kStrip));

    // ⌘L (Ctrl+L elsewhere) closes it, and the canvas takes the width back.
    ImGuiIO& io = ImGui::GetIO();
    const ImGuiKey mod = io.ConfigMacOSXBehaviors ? ImGuiMod_Super : ImGuiMod_Ctrl;
    io.AddKeyEvent(mod, true);
    io.AddKeyEvent(ImGuiKey_L, true);
    ui.frame([&] { pane.read_keys(); });
    io.AddKeyEvent(ImGuiKey_L, false);
    io.AddKeyEvent(mod, false);
    CHECK_FALSE(pane.open());
    ui.frame([&] {
        pane.draw_column();
        left = work_area();
    });
    CHECK(left.Max.x == doctest::Approx(Ui::kW));
    REQUIRE(read_chat_settings(path, &saved));
    CHECK_FALSE(saved.open);
}

TEST_CASE("chat pane: Ask in chat opens the pane with the handoff draft, which waits for the chat") {
    const fs::path dir = fresh("chat-pane-handoff");
    fs::create_directories(dir / "empty");
    const std::string path = (dir / "side-chat.json").string();
    REQUIRE(write_chat_settings(path, stub_settings()));
    App a(dir / "ws");
    ChatPane pane;
    ChatPane::Options o;
    o.settings_path = path;
    o.path = (dir / "empty").string();
    pane.start(o, {});
    REQUIRE_FALSE(pane.open());

    WidgetChatRequest r;
    r.request = "Build a loader for the validation logs";
    r.target = WidgetTarget::for_panel("curves");
    caliper_builder::ContextBundle bundle;
    bundle.blocks = json::array({{{"type", "text"}, {"text", "Context for a widget request about the panel curves."}}});
    chat_pane_handoff(pane, r, bundle, nullptr);
    CHECK(pane.open());
    CHECK(pane.draft_chips() == 0);   // the chat is made at the next frame
    REQUIRE(a.run(pane, [&] { return pane.draft_chips() > 0; }));
    SideChat* chat = chat_pane_chat(pane);
    REQUIRE(chat != nullptr);
    CHECK(chat->text() == r.request);
    CHECK(chat->draft().chips == std::vector<std::string>{"context", "ids"});
    CHECK(chat->items().empty());   // nothing is sent by itself
    CHECK(chat->turns() == 0);
}

TEST_CASE("chat pane: the agent's command that waits for approval is one row, and the API line and the end hook") {
    const fs::path dir = fresh("chat-pane-approval");
    fs::create_directories(dir / "empty");
    const fs::path api = dir / "api.json";
    std::ofstream(api) << "{}\n";   // the stub agent here does not call caliper-mcp
    const std::string path = (dir / "side-chat.json").string();
    // The chat stub, which ignores the opening context the pane sends first.
    fs::create_directories(dir / "gates");
    ChatPaneSettings s = stub_settings();
    s.chat.custom_command = std::string("'") + CALIPER_CHAT_STUB_AGENT + "' --gate-dir '" + (dir / "gates").string() + "'";
    s.open = true;
    REQUIRE(write_chat_settings(path, s));
    App a(dir / "ws");
    ChatPane pane;
    ChatPane::Options o;
    o.settings_path = path;
    o.path = (dir / "empty").string();
    int api_starts = 0, ended = 0;
    std::vector<std::string> pointed;
    ChatPane::Hooks h;
    // Each agent session gets an API file of its own, as AppApi issues them.
    std::vector<std::string> issued, ended_files;
    h.start_api = [&](std::string*, std::string* started) {
        if (api_starts++ == 0) *started = "127.0.0.1:61234";   // the first call started it
        const fs::path f = dir / ("api-agent-" + std::to_string(api_starts) + ".json");
        fs::copy_file(api, f, fs::copy_options::overwrite_existing);
        issued.push_back(f.string());
        return f.string();
    };
    h.end_api_file = [&](const std::string& f) { ended_files.push_back(f); };
    h.agent_ended = [&] { ++ended; };
    h.show_approval = [&](const std::string& id) { pointed.push_back(id); };
    pane.start(o, h);
    REQUIRE(a.run(pane, [&] { return pane.state() == "idle" && a.app.session(); }));
    caliper_builder::Session& session = *a.app.session();
    REQUIRE(invoke(session, "spec.new", {{"id", "dash"}, {"title", "Dash"}}).ok);

    SideChat* chat = chat_pane_chat(pane);
    REQUIRE(chat != nullptr);
    chat->set_text("hold");
    chat->send();
    REQUIRE(a.run(pane, [&] { return pane.state() == "running"; }));
    // The line the chat writes when the app started Caliper API for it.
    bool api_line = false;
    for (const ChatItem& it : chat->items())
        if (it.kind == ChatItem::Kind::notice && it.text == "API" && !it.lines.empty())
            api_line = it.lines[0] == "Caliper API started on 127.0.0.1:61234 for this chat";
    CHECK(api_line);

    // A panel.create from the origin agent, held for review.
    caliper_builder::CommandInvocation inv;
    inv.name = "panel.create";
    inv.args = {{"id", "note"}, {"primitive", "text"}, {"options", {{"text", "Hi"}}}};
    inv.origin = caliper_builder::CommandOrigin::agent;
    auto held = session.queue().submit(std::move(inv));
    REQUIRE(a.run(pane, [&] { return pane.has_item("approval", "panel.create", "waiting"); }));
    int rows = 0;
    std::string approval;
    for (const ChatItem& it : chat->items())
        if (it.kind == ChatItem::Kind::approval) {
            ++rows;
            approval = it.approval;
        }
    CHECK(rows == 1);
    REQUIRE_FALSE(approval.empty());
    session.queue().approve(approval);
    // Decided, the row says the outcome the queue recorded.
    REQUIRE(a.run(pane, [&] { return pane.has_item("approval", "panel.create", "approved"); }));
    REQUIRE(held.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    CHECK(held.get().ok);
    std::ofstream(dir / "gates" / "go").close();
    REQUIRE(a.run(pane, [&] { return pane.state() == "ready"; }));
    CHECK(ended == 0);   // the agent is still there

    // The agent ends: the app hears it, to stop an API it started for the chat.
    chat->set_text("Compare the first and the last layer");
    chat->send();
    REQUIRE(a.run(pane, [&] { return pane.state() == "exited"; }));
    CHECK(ended == 1);
    CHECK(api_starts == 1);
    // ...and the session's own API file, with its token, ends with it.
    CHECK(ended_files == issued);
    // A restart replaces the agent without an end, with a file of its own.
    chat->restart();
    REQUIRE(a.run(pane, [&] { return pane.state() == "ready"; }));
    CHECK(ended == 1);
    CHECK(api_starts == 2);
    REQUIRE(issued.size() == 2);
    CHECK(issued[1] != issued[0]);
    CHECK(ended_files.size() == 1);
    // The pane's shutdown ends the second.
    pane.shutdown();
    CHECK(ended_files == issued);
}
