// CPU-only: drive the real ImGui widgets and verify their wire requests.
// The peer is a tiny model substitute, so this never opens Metal or a model.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/live_inspector.h"
#include <crude_json.h>
#include <imgui_internal.h>
#include <implot.h>

#include <cerrno>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <memory>
#include <string>
#include <utility>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
using Json = crude_json::value;
using Object = crude_json::object;
using Array = crude_json::array;
constexpr const char* inference_window = "Inference###caliper.inspector.inference";
constexpr const char* tensor_window = "Tensor###caliper.inspector.tensor";

struct Peer {
    int listener = -1, socket = -1;
    std::string token, path, incoming;
    std::deque<Json> requests;

    Peer() {
        const auto directory = "/tmp/caliper-" + std::to_string(getuid());
        mkdir(directory.c_str(), 0700);
        token = "ui-" + std::to_string(getpid());
        path = directory + "/" + token + ".sock";
        listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
        REQUIRE(listener >= 0);
        sockaddr_un addr{}; addr.sun_family = AF_UNIX;
        std::strcpy(addr.sun_path, path.c_str());
        REQUIRE(bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
        REQUIRE(listen(listener, 1) == 0);
        fcntl(listener, F_SETFL, O_NONBLOCK);
    }
    ~Peer() {
        if (socket >= 0) ::close(socket);
        if (listener >= 0) ::close(listener);
        unlink(path.c_str());
    }
    void pump() {
        if (socket < 0) {
            socket = accept(listener, nullptr, nullptr);
            if (socket < 0) return;
            fcntl(socket, F_SETFL, O_NONBLOCK);
        }
        char bytes[8192];
        for (;;) {
            const auto n = recv(socket, bytes, sizeof bytes, 0);
            if (n <= 0) break;
            incoming.append(bytes, static_cast<size_t>(n));
        }
        size_t end;
        while ((end = incoming.find('\n')) != std::string::npos) {
            requests.push_back(Json::parse(incoming.substr(0, end)));
            incoming.erase(0, end + 1);
        }
    }
    void send(const Json& message) {
        const std::string bytes = message.dump() + "\n";
        REQUIRE(::send(socket, bytes.data(), bytes.size(), 0) == static_cast<ssize_t>(bytes.size()));
    }
    void reply(const Json& request, Json result) {
        send(Object{{"id", request["id"]}, {"done", true}, {"result", std::move(result)}});
    }
    void disconnect() {
        if (socket >= 0) ::close(socket);
        socket = -1;
    }
};

struct Ui {
    Peer peer;
    std::unique_ptr<caliper_live::Inspector> inspector;
    std::string captured;
    int errors = 0;

    Ui() {
        ImGui::CreateContext();
        ImPlot::CreateContext();
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.DisplaySize = ImVec2(1600, 1400);
        io.DeltaTime = 1.0f / 60;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
        io.ConfigErrorRecoveryEnableAssert = false;
        io.ConfigErrorRecoveryEnableTooltip = false;
        io.ConfigErrorRecoveryEnableDebugLog = false;
        ImGui::GetCurrentContext()->ErrorCallback = [](ImGuiContext*, void* userdata, const char*) {
            ++static_cast<Ui*>(userdata)->errors;
        };
        ImGui::GetCurrentContext()->ErrorCallbackUserData = this;
        inspector = std::make_unique<caliper_live::Inspector>();
    }
    ~Ui() {
        inspector.reset();
        if (ImGui::GetCurrentContext()->WithinFrameScope) ImGui::EndFrame();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
    void frame() {
        inspector->poll(peer.token, true);
        ImGui::NewFrame();
        // Capture only in memory; no ini, clipboard or log files are written.
        ImGui::LogToBuffer();
        inspector->draw(0, 0, 1600, 1400);
        captured = ImGui::GetCurrentContext()->LogBuffer.c_str();
        ImGui::LogFinish();
        ImGui::Render();
        for (auto* tex : ImGui::GetPlatformIO().Textures) {
            if (tex->Status == ImTextureStatus_WantCreate || tex->Status == ImTextureStatus_WantUpdates) {
                tex->SetTexID((ImTextureID)1);
                tex->SetStatus(ImTextureStatus_OK);
            } else if (tex->Status == ImTextureStatus_WantDestroy) {
                tex->SetTexID(ImTextureID_Invalid);
                tex->SetStatus(ImTextureStatus_Destroyed);
            }
        }
        peer.pump();
    }
    Json request(const char* op) {
        for (int i = 0; i < 30 && peer.requests.empty(); ++i) frame();
        REQUIRE_FALSE(peer.requests.empty());
        Json result = std::move(peer.requests.front()); peer.requests.pop_front();
        REQUIRE(result.is_object());
        CHECK(result["op"].get<std::string>() == op);
        return result;
    }
    void activate(const char* window_name, const char* label) {
        ImGuiWindow* window = ImGui::FindWindowByName(window_name);
        REQUIRE(window != nullptr);
        ImGui::FocusWindow(window);
        ImGui::ActivateItemByID(window->GetID(label));
        frame();
    }
    void input(const char* window_name, const char* label, const char* text) {
        activate(window_name, label);
        ImGui::GetIO().AddInputCharactersUTF8(text); // numeric inputs select all on activation
        frame();
        ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, true);
        frame();
        ImGui::GetIO().AddKeyEvent(ImGuiKey_Enter, false);
        frame();
    }
    void overview(const Json& req, bool edited = false) {
        Array edits;
        if (edited) edits.push_back(Object{{"id", 1.0}, {"name", "block.weight"}, {"description", "Set (0, 0) to 0.5"}});
        peer.reply(req, Object{{"name", "block.weight"}, {"rows", 4.0}, {"cols", 3.0},
            {"sample_rows", 2.0}, {"sample_cols", 2.0},
            {"values", Array{0.125, 0.25, -0.5, 1.0}},
            {"row_indices", Array{0.0, 3.0}}, {"col_indices", Array{0.0, 2.0}},
            {"min", -0.5}, {"max", 1.0}, {"edits", std::move(edits)}});
        frame();
    }
    void connect() {
        auto hello = request("hello");
        peer.reply(hello, Object{{"protocol", 1.0}, {"capabilities", Array{
            "generate", "list_tensors", "show_tensor", "read_weight", "edit_weight", "revert_tensor"}}});
        auto list = request("list_tensors");
        peer.reply(list, Object{{"model", "Test model"}, {"tensors", Array{
            Object{{"name", "block.weight"}, {"group", "Block"}, {"shape", Array{4.0, 3.0}},
                {"dtype", "float32"}, {"bytes", 48.0}, {"size", 12.0}, {"rows", 4.0}, {"cols", 3.0}, {"tied", true}}
        }}});
        overview(request("show_tensor"));
        frame();
    }
};
} // namespace

TEST_CASE("inspector widgets stream, stop and edit through the request channel") {
    Ui ui;
    ui.connect();
    CHECK(ui.captured.find("Test model") != std::string::npos);
    CHECK(ui.captured.find("Shared with the output layer") != std::string::npos);
    CHECK(ui.captured.find("Overview: 2 x 2 sampled values") != std::string::npos);
    CHECK(ui.captured.find("Max new tokens") != std::string::npos);

    ui.activate(inference_window, "Generate");
    const auto generation = ui.request("generate");
    CHECK(generation["prompt"].get<std::string>() == "Write one sentence about the Moon.");
    CHECK(generation["max_new_tokens"].get<double>() == 128);
    CHECK(generation["thinking"].get<bool>() == false);
    ui.peer.send(Object{{"id", generation["id"]}, {"done", false}, {"event", Object{
        {"text", "A streamed answer."}, {"tokens", 4.0}, {"tokens_per_second", 22.5}}}});
    ui.frame();
    CHECK(ui.captured.find("A streamed answer.") != std::string::npos);
    CHECK(ui.captured.find("22.5 tokens/s") != std::string::npos);

    // Actual disabled ImGui widgets cannot emit a mutation during generation.
    ui.activate(tensor_window, "Zero row");
    ui.frame();
    CHECK(ui.peer.requests.empty());
    ui.activate(inference_window, "Stop");
    const auto stop = ui.request("stop");
    ui.peer.reply(stop, Object{{"stopping", true}});
    ui.peer.reply(generation, Object{{"tokens", 4.0}, {"tokens_per_second", 22.5}, {"cancelled", true}});
    ui.frame();
    CHECK(ui.captured.find("Stopped") != std::string::npos);

    ui.activate(tensor_window, "Read");
    const auto read = ui.request("read_weight");
    CHECK(read["name"].get<std::string>() == "block.weight");
    CHECK(read["row"].get<double>() == 0);
    CHECK(read["col"].get<double>() == 0);
    ui.peer.reply(read, Object{{"name", "block.weight"}, {"row", 0.0}, {"col", 0.0}, {"value", 0.125}});
    ui.frame();
    CHECK(ui.captured.find("(0, 0) = 0.125") != std::string::npos);
    ui.input(tensor_window, "Value", "0.5");
    ui.activate(tensor_window, "Set");
    auto edit = ui.request("edit_weight");
    CHECK(edit["action"].get<std::string>() == "set");
    CHECK(edit["value"].get<double>() == 0.5);
    ui.peer.reply(edit, Object{{"name", "block.weight"}, {"edits", Array{}}});
    ui.overview(ui.request("show_tensor"), true);
    CHECK(ui.captured.find("Set (0, 0) to 0.5") != std::string::npos);

    for (const auto& action : {std::make_pair("Zero row", "zero_row"), {"Zero column", "zero_column"},
                              {"Scale row", "scale_row"}, {"Scale column", "scale_column"}}) {
        ui.activate(tensor_window, action.first);
        edit = ui.request("edit_weight");
        CHECK(edit["action"].get<std::string>() == action.second);
        ui.peer.reply(edit, Object{{"name", "block.weight"}, {"edits", Array{}}});
        ui.overview(ui.request("show_tensor"));
    }
    ui.activate(tensor_window, "Revert tensor");
    const auto revert = ui.request("revert_tensor");
    CHECK(revert["name"].get<std::string>() == "block.weight");
    ui.peer.reply(revert, Object{{"name", "block.weight"}, {"edits", Array{}}});
    ui.overview(ui.request("show_tensor"));
    CHECK(ui.captured.find("No edits") != std::string::npos);
    CHECK(ui.errors == 0);
}

TEST_CASE("inspector reports disconnect without replaying generation") {
    Ui ui;
    ui.connect();
    ui.activate(inference_window, "Generate");
    ui.request("generate");
    ui.peer.disconnect();
    ui.frame();
    CHECK(ui.captured.find("Generation interrupted") != std::string::npos);
    CHECK(ui.captured.find("operation was not retried") != std::string::npos);
    ui.activate(inference_window, "Generate");
    ui.frame();
    CHECK(ui.peer.requests.empty());
    CHECK(ui.errors == 0);
}
