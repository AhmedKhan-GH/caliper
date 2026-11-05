// CPU-only: drive the real ImGui widgets and verify their wire requests.
// The peer is a tiny model substitute, so this never opens Metal or a model.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "live/live_inspector.h"
#include <crude_json.h>
#include <imgui_internal.h>
#include <implot.h>

#include <algorithm>
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
    std::string token, path, incoming, outgoing;
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
    void flush() {
        while (socket >= 0 && !outgoing.empty()) {
            const auto n = ::send(socket, outgoing.data(), outgoing.size(), 0);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
            REQUIRE(n > 0);
            outgoing.erase(0, static_cast<size_t>(n));
        }
    }
    void pump() {
        if (socket < 0) {
            socket = accept(listener, nullptr, nullptr);
            if (socket < 0) return;
            fcntl(socket, F_SETFL, O_NONBLOCK);
        }
        flush();
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
        outgoing += message.dump() + "\n";
        flush();
    }
    void reply(const Json& request, Json result) {
        send(Object{{"id", request["id"]}, {"done", true}, {"result", std::move(result)}});
    }
    void disconnect() {
        if (socket >= 0) ::close(socket);
        socket = -1;
        outgoing.clear();
    }
};

struct Ui {
    Peer peer;
    std::unique_ptr<caliper_live::Inspector> inspector;
    std::string captured;
    int errors = 0;
    Json initial_sample;
    bool vector_fixture = false, embedding_fixture = false;

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
        for (int tries = 0; tries < 100; ++tries) {
            for (int i = 0; i < 30 && peer.requests.empty(); ++i) frame();
            REQUIRE_FALSE(peer.requests.empty());
            Json result = std::move(peer.requests.front()); peer.requests.pop_front();
            REQUIRE(result.is_object());
            const auto received = result["op"].get<std::string>();
            // Lazy previews may be outstanding when a manual control is used.
            // Complete those tiny reads and observe the queued action next.
            if (received == "show_tensor" && std::string(op) != "show_tensor") {
                overview(result); continue;
            }
            CHECK(received == op);
            return result;
        }
        FAIL("request did not reach the front of the serial scheduler");
        return {};
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
        if (vector_fixture) {
            const Json& region = req.contains("region") ? req["region"] : Json(Object{
                {"row", 0.0}, {"col", 0.0}, {"rows", 1024.0}, {"cols", 1.0}});
            const int first = static_cast<int>(region["row"].get<double>());
            const int count = static_cast<int>(region["rows"].get<double>());
            const int target = req.contains("target") ? static_cast<int>(req["target"]["height"].get<double>()) : 128;
            const int samples = std::min(count, target);
            Array indices, values;
            for (int i = 0; i < samples; ++i) {
                const int index = first + (samples == 1 ? 0 : i * (count - 1) / (samples - 1));
                indices.push_back(static_cast<double>(index)); values.push_back(0.125 + index * 0.001);
            }
            peer.reply(req, Object{{"name", "block.weight"}, {"rows", 1024.0}, {"cols", 1.0},
                {"sample_rows", static_cast<double>(samples)}, {"sample_cols", 1.0},
                {"values", std::move(values)}, {"row_indices", std::move(indices)}, {"col_indices", Array{0.0}},
                {"region", region}, {"edited", edited}, {"revision", edited ? 1.0 : 0.0}, {"edits", std::move(edits)}});
        } else if (embedding_fixture) {
            const Json& region = req["region"];
            peer.reply(req, Object{{"name", "block.weight"}, {"rows", 248320.0}, {"cols", 1024.0},
                {"sample_rows", 1.0}, {"sample_cols", 1.0}, {"values", Array{0.25}},
                {"row_indices", Array{region["row"]}}, {"col_indices", Array{region["col"]}},
                {"region", region}, {"edited", false}, {"revision", 0.0}, {"edits", Array{}}});
        } else {
            peer.reply(req, Object{{"name", "block.weight"}, {"rows", 4.0}, {"cols", 3.0},
                {"sample_rows", 2.0}, {"sample_cols", 2.0},
                {"values", Array{0.125, 0.25, -0.5, 1.0}},
                {"row_indices", Array{0.0, 3.0}}, {"col_indices", Array{0.0, 2.0}},
                {"min", -0.5}, {"max", 1.0}, {"edits", std::move(edits)},
                {"region", Object{{"row", 0.0}, {"col", 0.0}, {"rows", 4.0}, {"cols", 3.0}}},
                {"edited", edited}, {"revision", edited ? 1.0 : 0.0}});
        }
        frame();
        for (int i = 0; i < 30 && !peer.outgoing.empty(); ++i) frame();
        REQUIRE(peer.outgoing.empty());
        frame(); // Consume the final chunk after the peer flushed it.
    }
    void connect(bool regions = false, bool vector = false, bool embedding = false) {
        vector_fixture = vector; embedding_fixture = embedding;
        auto hello = request("hello");
        Array capabilities{"generate", "list_tensors", "show_tensor", "read_weight", "edit_weight", "revert_tensor"};
        if (regions) capabilities.push_back("tensor_regions");
        peer.reply(hello, Object{{"protocol", 1.0}, {"capabilities", std::move(capabilities)}});
        auto list = request("list_tensors");
        peer.reply(list, Object{{"model", "Test model"}, {"tensors", Array{
            Object{{"name", "block.weight"}, {"group", "Block"},
                {"shape", vector ? Array{1024.0} : embedding ? Array{248320.0, 1024.0} : Array{4.0, 3.0}},
                {"dtype", "float32"}, {"bytes", vector ? 4096.0 : embedding ? 1017118720.0 : 48.0},
                {"size", vector ? 1024.0 : embedding ? 254279680.0 : 12.0},
                {"rows", vector ? 1024.0 : embedding ? 248320.0 : 4.0},
                {"cols", vector ? 1.0 : embedding ? 1024.0 : 3.0}, {"tied", true},
                {"role", embedding ? "embedding" : "weight"}}
        }}});
        initial_sample = request("show_tensor");
        overview(initial_sample);
        frame();
    }
};
} // namespace

TEST_CASE("inspector widgets stream, stop and edit through the request channel") {
    Ui ui;
    ui.connect();
    CHECK(ui.captured.find("Test model") != std::string::npos);
    CHECK(ui.captured.find("Shared with the output layer") != std::string::npos);
    CHECK(ui.captured.find("Region: 2 x 2 sampled values") != std::string::npos);
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
    ui.input(tensor_window, "##value", "0.5");   // Enter in the Value field sets the weight
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

TEST_CASE("inspector requests visible region pixels and prioritizes manual generation") {
    Ui ui;
    ui.connect(true);
    const auto& region = ui.initial_sample["region"];
    CHECK(region["row"].get<double>() == 0);
    CHECK(region["col"].get<double>() == 0);
    CHECK(region["rows"].get<double>() == 4);
    CHECK(region["cols"].get<double>() == 3);
    const auto& target = ui.initial_sample["target"];
    CHECK(target["width"].get<double>() == 3);
    CHECK(target["height"].get<double>() == 4);
    CHECK(ui.captured.find("Overview only") == std::string::npos);
    CHECK(ui.captured.find("Reset layout") != std::string::npos);

    // Leave a passive overview/thumbnail in flight. Generate stays available,
    // and its request is issued immediately when that passive read completes.
    const auto passive = ui.request("show_tensor");
    ui.activate(inference_window, "Generate");
    CHECK(ui.peer.requests.empty());
    ui.overview(passive);
    const auto generation = ui.request("generate");
    CHECK(generation["prompt"].is_string());
    ui.peer.reply(generation, Object{{"tokens", 1.0}, {"tokens_per_second", 5.0}, {"cancelled", false}});
    ui.frame();
    CHECK(ui.captured.find("Complete") != std::string::npos);
    CHECK(ui.errors == 0);
}

TEST_CASE("inspector vector detail uses square strip cells and a separate full-range plot") {
    Ui ui;
    ui.connect(true, true);
    const double visible = ui.initial_sample["region"]["rows"].get<double>();
    CHECK(visible > 1);
    CHECK(visible < 1024);
    CHECK(ui.initial_sample["region"]["cols"].get<double>() == 1);
    CHECK(ui.initial_sample["target"]["width"].get<double>() == 1);
    CHECK(ui.initial_sample["target"]["height"].get<double>() == visible);

    const auto plot = ui.request("show_tensor");
    CHECK(plot["region"]["row"].get<double>() == 0);
    CHECK(plot["region"]["rows"].get<double>() == 1024);
    CHECK(plot["region"]["cols"].get<double>() == 1);
    CHECK(plot["target"]["width"].get<double>() == 1);
    CHECK(plot["target"]["height"].get<double>() > visible);
    ui.overview(plot);
    CHECK(ui.captured.find("exact cells") != std::string::npos);
    CHECK(ui.errors == 0);
}

TEST_CASE("inspector requests folded embedding thumbnails within the small atlas budget") {
    Ui ui;
    ui.connect(true, false, true);
    const auto whole_overview = ui.request("show_tensor");
    CHECK(whole_overview["region"]["rows"].get<double>() == 248320);
    CHECK(whole_overview["region"]["cols"].get<double>() == 1024);
    ui.overview(whole_overview);

    const auto thumbnail = ui.request("show_tensor");
    CHECK(thumbnail["name"].get<std::string>() == "block.weight");
    CHECK(thumbnail["region"]["row"].get<double>() == 0);
    CHECK(thumbnail["region"]["col"].get<double>() == 0);
    CHECK(thumbnail["region"]["rows"].get<double>() == 248320);
    CHECK(thumbnail["region"]["cols"].get<double>() == 1024);
    // Sixteen equal row blocks become a 128 x 128 texture after reflow.
    CHECK(thumbnail["target"]["height"].get<double>() == 2048);
    CHECK(thumbnail["target"]["width"].get<double>() == 8);
    CHECK(ui.errors == 0);
}

namespace {
std::string base64(const std::string& bytes) {
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t n = static_cast<uint8_t>(bytes[i]) << 16;
        if (i + 1 < bytes.size()) n |= static_cast<uint8_t>(bytes[i + 1]) << 8;
        if (i + 2 < bytes.size()) n |= static_cast<uint8_t>(bytes[i + 2]);
        out += alphabet[n >> 18 & 63]; out += alphabet[n >> 12 & 63];
        out += i + 1 < bytes.size() ? alphabet[n >> 6 & 63] : '=';
        out += i + 2 < bytes.size() ? alphabet[n & 63] : '=';
    }
    return out;
}
// Rows of a 4 x 3 tensor as bfloat16 bits: the top half of each float32.
std::string bfloat16_rows(int first, int count) {
    std::string bytes;
    for (int i = first * 3; i < (first + count) * 3; ++i) {
        const float value = 0.25f * static_cast<float>(i) - 1.0f;
        uint32_t bits; std::memcpy(&bits, &value, sizeof bits);
        bytes += static_cast<char>(bits >> 16 & 255); bytes += static_cast<char>(bits >> 24 & 255);
    }
    return base64(bytes);
}
} // namespace

TEST_CASE("inspector fetches whole values once, then zooms and selects without the host") {
    Ui ui;
    auto hello = ui.request("hello");
    ui.peer.reply(hello, Object{{"protocol", 1.0}, {"capabilities", Array{"list_tensors", "show_tensor", "read_weight",
        "edit_weight", "tensor_regions", "tensor_values"}}});
    auto list = ui.request("list_tensors");
    ui.peer.reply(list, Object{{"model", "Test model"}, {"tensors", Array{
        Object{{"name", "block.weight"}, {"group", "Block"}, {"shape", Array{4.0, 3.0}}, {"dtype", "bfloat16"},
               {"bytes", 24.0}, {"size", 12.0}, {"rows", 4.0}, {"cols", 3.0}}}}});
    // Whole rows, each reply continuing where the last ended.
    auto first = ui.request("tensor_values");
    CHECK(first["row"].get<double>() == 0);
    ui.peer.reply(first, Object{{"name", "block.weight"}, {"rows", 4.0}, {"cols", 3.0}, {"row", 0.0}, {"count", 3.0},
        {"encoding", "bfloat16"}, {"data", bfloat16_rows(0, 3)}, {"revision", 0.0}});
    auto second = ui.request("tensor_values");
    CHECK(second["row"].get<double>() == 3);
    ui.peer.reply(second, Object{{"name", "block.weight"}, {"rows", 4.0}, {"cols", 3.0}, {"row", 3.0}, {"count", 1.0},
        {"encoding", "bfloat16"}, {"data", bfloat16_rows(3, 1)}, {"revision", 0.0}});
    for (int i = 0; i < 5; ++i) ui.frame();
    CHECK(ui.captured.find("Zoom 1x") != std::string::npos);

    // Zoom and click in the picture: the view and the value come from the
    // values held here. The only request is the 3D view's thumbnail.
    ImGuiWindow* window = ImGui::FindWindowByName(tensor_window);
    REQUIRE(window != nullptr);
    auto& io = ImGui::GetIO();
    const ImVec2 at(window->Pos.x + window->Size.x * 0.45f, window->Pos.y + window->Size.y * 0.4f);
    io.AddMousePosEvent(at.x, at.y);
    ui.frame();
    io.AddMouseWheelEvent(0, 1);
    ui.frame(); ui.frame();
    // Zoomed in (a 4 x 3 tensor stops at max_cell), the hovered cell's exact value decoded from bfloat16.
    CHECK(ui.captured.find("Zoom 1x") == std::string::npos);
    CHECK(ui.captured.find("Zoom 1.") != std::string::npos);
    CHECK(ui.captured.find("(1, 1)\n0\n") != std::string::npos);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); ui.frame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); ui.frame();
    int region_requests = 0;
    for (int i = 0; i < 30; ++i) {
        ui.frame();
        while (!ui.peer.requests.empty()) {
            Json request = std::move(ui.peer.requests.front()); ui.peer.requests.pop_front();
            const auto op = request["op"].get<std::string>();
            CHECK(op == "show_tensor");
            ++region_requests;
            ui.overview(request);
        }
    }
    CHECK(region_requests <= 1);
    CHECK(ui.captured.find("(1, 1) = 0") != std::string::npos);
    CHECK(ui.captured.find("Value not read") == std::string::npos);
    CHECK(ui.errors == 0);
}

#include "live/live_trace.h"

namespace {
constexpr const char* trace_window = "Trace###caliper.inspector.trace";
// One trace event: positions first.. with tokens, two layers and two words per cell. Words: id n is "w<n>".
Json trace_event(int first, std::vector<int> tokens, int prompt, int top = 7) {
    Array ids, jlens, logit, output, probs, cell_probs, out_probs;
    Object words;
    for (int t : tokens) { ids.push_back(static_cast<double>(t)); words[std::to_string(t)] = " w" + std::to_string(t); }
    for (size_t p = 0; p < tokens.size(); ++p) {
        for (int layer = 0; layer < 2; ++layer)
            for (int rank = 0; rank < 2; ++rank) {
                const int id = top + layer + rank * 10;
                jlens.push_back(static_cast<double>(id)); logit.push_back(static_cast<double>(id + 1));
                cell_probs.push_back(rank ? 0.1 : 0.6);
                words[std::to_string(id)] = " w" + std::to_string(id);
                words[std::to_string(id + 1)] = " w" + std::to_string(id + 1);
            }
        output.push_back(static_cast<double>(top)); output.push_back(static_cast<double>(top + 20));
        out_probs.push_back(0.7); out_probs.push_back(0.2);
        words[std::to_string(top + 20)] = " w" + std::to_string(top + 20);
    }
    return Object{{"first", static_cast<double>(first)}, {"count", static_cast<double>(tokens.size())},
                  {"prompt", static_cast<double>(prompt)}, {"layers", Array{0.0, 1.0}}, {"k", 2.0},
                  {"tokens", ids}, {"jlens", jlens}, {"jprob", cell_probs}, {"logit", logit}, {"lprob", cell_probs},
                  {"output", output}, {"oprob", out_probs}, {"words", words}};
}
} // namespace

TEST_CASE("trace events extend a run in order and are refused when they do not fit") {
    caliper_live::TraceRun run;
    std::map<int32_t, std::string> words;
    std::string error;
    REQUIRE(caliper_live::read_trace_event(trace_event(0, {1, 2}, 2), run, words, error));
    REQUIRE(caliper_live::read_trace_event(trace_event(2, {3}, 2), run, words, error));
    CHECK(run.positions() == 3); CHECK(run.prompt == 2); CHECK(run.generated() == 1);
    CHECK(run.jlens[run.at(2, 1, 1)] == 18); CHECK(words.at(18) == " w18");
    CHECK_FALSE(caliper_live::read_trace_event(trace_event(5, {4}, 2), run, words, error));   // a gap
    Json wrong = trace_event(3, {4}, 2);
    wrong["jlens"] = Array{1.0};
    CHECK_FALSE(caliper_live::read_trace_event(wrong, run, words, error));                      // a short array
    CHECK(run.positions() == 3);
    // Attention for two layers and two heads over three keys: per position, layer, head, key.
    caliper_live::TraceRun attended;
    Json with_attention = trace_event(0, {1}, 1);
    std::string bytes;
    for (int v : {255, 0, 0, 51, 204, 0, 0, 255, 0, 128, 127, 0}) bytes += static_cast<char>(v);
    with_attention["attention_layers"] = Array{3.0, 7.0};
    with_attention["heads"] = 2.0;
    with_attention["keys"] = 3.0;
    with_attention["attention"] = base64(bytes);
    REQUIRE(caliper_live::read_trace_event(with_attention, attended, words, error));
    CHECK(attended.weight(0, 0, 0, 0) == doctest::Approx(1.0));
    CHECK(attended.weight(0, 0, 1, 1) == doctest::Approx(0.8));
    CHECK(attended.weight(0, 0, 0) == doctest::Approx((255 + 51) / 510.0));      // the mean of the heads
    CHECK(attended.weight(0, 1, 1) == doctest::Approx((255 + 127) / 510.0));
    CHECK(attended.weight(0, 1, 5) == 0);
    with_attention = trace_event(1, {2}, 1);
    with_attention["attention_layers"] = Array{3.0, 7.0};
    with_attention["heads"] = 2.0;
    with_attention["keys"] = 3.0;
    with_attention["attention"] = base64(bytes.substr(0, 9));                     // short
    CHECK_FALSE(caliper_live::read_trace_event(with_attention, attended, words, error));
    CHECK(caliper_live::display_word(" Paris") == "Paris");
    CHECK(caliper_live::display_word("\n\n") == "\\n\\n");
    CHECK(caliper_live::display_word(" ") == "\xC2\xB7");
}

TEST_CASE("the Trace pane traces a prompt, then compares an intact run with an intervened one") {
    Ui ui;
    auto hello = ui.request("hello");
    ui.peer.reply(hello, Object{{"protocol", 1.0}, {"capabilities", Array{"generate", "list_tensors", "show_tensor", "trace", "tokenize"}}});
    auto list = ui.request("list_tensors");
    ui.peer.reply(list, Object{{"model", "Test model"}, {"tensors", Array{
        Object{{"name", "block.weight"}, {"group", "Block"}, {"shape", Array{4.0, 3.0}}, {"dtype", "float32"},
               {"bytes", 48.0}, {"size", 12.0}, {"rows", 4.0}, {"cols", 3.0}}}}});
    for (int i = 0; i < 5; ++i) ui.frame();
    // Trace takes Model 3D's place through the switch in its toolbar, and gives it back.
    CHECK(ui.captured.find("Fit Focus selected Trace") != std::string::npos);
    ui.activate("Model 3D###caliper.inspector.model3d", "Trace");
    ui.frame();
    REQUIRE(ImGui::FindWindowByName(trace_window) != nullptr);
    CHECK(ui.captured.find("Fit Focus selected") == std::string::npos);
    // Generate in the Inference pane runs a trace with its prompt and sampling, attention included.
    ui.activate(inference_window, "Generate");
    auto plain = ui.request("trace");
    CHECK(plain["interventions"].get<Array>().empty());
    CHECK(plain["prompt"].get<std::string>() == "Write one sentence about the Moon.");
    CHECK(plain["temperature"].get<double>() == doctest::Approx(0.7));
    CHECK(plain["attention"].get<bool>());
    ui.peer.send(Object{{"id", plain["id"]}, {"done", false}, {"event", trace_event(0, {1, 2}, 2)}});
    ui.peer.send(Object{{"id", plain["id"]}, {"done", false}, {"event", trace_event(2, {3}, 2)}});
    ui.peer.reply(plain, Object{{"tokens", 1.0}, {"finish_reason", "length"}, {"cancelled", false}});
    for (int i = 0; i < 4; ++i) ui.frame();
    CHECK(ui.captured.find("Intact  w3") != std::string::npos);            // the answer: the generated token
    CHECK(ui.captured.find("Output\n w3") != std::string::npos);          // and the Inference pane's output
    CHECK(ui.captured.find("Position 2 w3") != std::string::npos);        // the side list follows the newest token
    CHECK(ui.captured.find("w7") != std::string::npos);

    // A J-space ablation, compared with the intact run: two traces, the second carrying the intervention.
    ui.activate(trace_window, "Add");
    ui.frame();
    // The default band (layers 5 to 17) is clamped to this two-layer model.
    CHECK(ui.captured.find("Ablate J-space, top 10  layers 1 to 1") != std::string::npos);
    ui.activate(inference_window, "Generate");
    auto intact = ui.request("trace");
    CHECK(intact["interventions"].get<Array>().empty());
    ui.peer.send(Object{{"id", intact["id"]}, {"done", false}, {"event", trace_event(0, {1, 2, 3}, 2)}});
    ui.peer.reply(intact, Object{{"tokens", 1.0}, {"finish_reason", "length"}});
    auto intervened = ui.request("trace");
    REQUIRE(intervened["interventions"].get<Array>().size() == 1);
    CHECK(intervened["interventions"][0]["kind"].get<std::string>() == "ablate");
    CHECK(intervened["interventions"][0]["k"].get<double>() == 10);
    ui.peer.send(Object{{"id", intervened["id"]}, {"done", false}, {"event", trace_event(0, {1, 2, 4}, 2, 9)}});
    ui.peer.reply(intervened, Object{{"tokens", 1.0}, {"finish_reason", "length"}});
    for (int i = 0; i < 4; ++i) ui.frame();
    CHECK(ui.captured.find("Intact  w3") != std::string::npos);
    CHECK(ui.captured.find("Intervened  w4") != std::string::npos);
    CHECK(ui.captured.find("w9") != std::string::npos);                   // the intervened run's words are shown
    CHECK(ui.errors == 0);
}

TEST_CASE("replies with UTF-8 text and unicode escapes parse to UTF-8 strings") {
    // Raw UTF-8 bytes (the script sends ensure_ascii=False), escapes in and beyond the basic plane, and a
    // lone high surrogate followed by an ordinary escape.
    const Json reply = Json::parse(
        "{\"a\":\"caf\xC3\xA9 \xE5\x9C\xB0\",\"b\":\"\\u00e9\\u5730\",\"c\":\"\\ud83d\\ude00\",\"d\":\"\\ud83dx\\n\"}");
    REQUIRE(reply.is_object());
    CHECK(reply["a"].get<std::string>() == "caf\xC3\xA9 \xE5\x9C\xB0");
    CHECK(reply["b"].get<std::string>() == "\xC3\xA9\xE5\x9C\xB0");
    CHECK(reply["c"].get<std::string>() == "\xF0\x9F\x98\x80");
    CHECK(reply["d"].get<std::string>() == "\xED\xA0\xBDx\n");
}

TEST_CASE("the inspector asks for binary samples when offered and draws them") {
    Ui ui;
    auto hello = ui.request("hello");
    ui.peer.reply(hello, Object{{"protocol", 1.0}, {"capabilities", Array{"list_tensors", "show_tensor",
        "tensor_regions", "sample_data"}}});
    auto list = ui.request("list_tensors");
    ui.peer.reply(list, Object{{"model", "Test model"}, {"tensors", Array{
        Object{{"name", "block.weight"}, {"group", "Block"}, {"shape", Array{4.0, 3.0}}, {"dtype", "float32"},
               {"bytes", 48.0}, {"size", 12.0}, {"rows", 4.0}, {"cols", 3.0}}}}});
    auto sample = ui.request("show_tensor");
    CHECK(sample["encoding"].get<std::string>() == "float32");
    // Two by two values, little-endian float32: 0.125, 0.25, -0.5, 1.
    std::string bytes;
    for (float v : {0.125f, 0.25f, -0.5f, 1.0f}) { char b[4]; std::memcpy(b, &v, 4); bytes.append(b, 4); }
    ui.peer.reply(sample, Object{{"name", "block.weight"}, {"rows", 4.0}, {"cols", 3.0}, {"sample_rows", 2.0},
        {"sample_cols", 2.0}, {"data", base64(bytes)}, {"row_indices", Array{0.0, 3.0}}, {"col_indices", Array{0.0, 2.0}},
        {"region", Object{{"row", 0.0}, {"col", 0.0}, {"rows", 4.0}, {"cols", 3.0}}}, {"edited", false}, {"revision", 0.0}});
    for (int i = 0; i < 4; ++i) ui.frame();
    CHECK(ui.captured.find("Region: 2 x 2 sampled values") != std::string::npos);
    CHECK(ui.errors == 0);
}

TEST_CASE("the playback clock shows the prompt's layers together, then each token from layer 0 to the output") {
    using caliper_live::playback_shows;
    using caliper_live::playback_positions;
    // Two lens layers plus the output row: three steps a token. A prompt of four tokens, then two generated.
    CHECK(playback_positions(0, 2, 4, 6) == 4);                       // the prompt's tokens show from the start
    CHECK_FALSE(playback_shows(0, 2, 4, 0, 0));
    CHECK(playback_shows(1, 2, 4, 3, 0));                             // step 1: layer 0 at every prompt position
    CHECK_FALSE(playback_shows(1, 2, 4, 3, 1));
    CHECK(playback_shows(3, 2, 4, 0, -1));                            // step 3: the prompt's output row
    CHECK(playback_positions(3, 2, 4, 6) == 4);
    CHECK(playback_positions(4, 2, 4, 6) == 5);                       // step 4: the first generated token enters
    CHECK(playback_shows(4, 2, 4, 4, 0));
    CHECK_FALSE(playback_shows(4.9, 2, 4, 4, 1));                     // fractions of a step show nothing more
    CHECK(playback_shows(6, 2, 4, 4, -1));
    CHECK_FALSE(playback_shows(6, 2, 4, 5, 0));
    CHECK(playback_positions(1e18, 2, 4, 6) == 6);                    // full speed: everything
    CHECK(playback_shows(1e18, 2, 4, 5, -1));
}

TEST_CASE("slowed down, the Trace pane and the Output reveal a finished generation a layer at a time") {
    Ui ui;
    auto hello = ui.request("hello");
    ui.peer.reply(hello, Object{{"protocol", 1.0}, {"capabilities", Array{"generate", "list_tensors", "show_tensor", "trace"}}});
    auto list = ui.request("list_tensors");
    ui.peer.reply(list, Object{{"model", "Test model"}, {"tensors", Array{
        Object{{"name", "block.weight"}, {"group", "Block"}, {"shape", Array{4.0, 3.0}}, {"dtype", "float32"},
               {"bytes", 48.0}, {"size", 12.0}, {"rows", 4.0}, {"cols", 3.0}}}}});
    for (int i = 0; i < 3; ++i) ui.frame();
    ui.activate("Model 3D###caliper.inspector.model3d", "Trace");
    ui.frame();
    // The speed slider, moved with the keyboard to its slow end (Shift steps a tenth of the range).
    ImGuiWindow* trace = ImGui::FindWindowByName(trace_window);
    REQUIRE(trace != nullptr);
    ImGui::FocusWindow(trace);
    ImGui::ActivateItemByID(trace->GetID("##trace.speed"));
    ui.frame();
    auto& io = ImGui::GetIO();
    for (int i = 0; i < 14; ++i) {
        io.AddKeyEvent(ImGuiMod_Shift, true); io.AddKeyEvent(ImGuiKey_LeftArrow, true); ui.frame();
        io.AddKeyEvent(ImGuiKey_LeftArrow, false); io.AddKeyEvent(ImGuiMod_Shift, false); ui.frame();
    }
    io.AddKeyEvent(ImGuiKey_Enter, true); ui.frame(); io.AddKeyEvent(ImGuiKey_Enter, false); ui.frame();
    CHECK(ui.captured.find("s per layer") != std::string::npos);
    // A whole generation arrives at once: two prompt tokens, then three generated ones.
    ui.activate(inference_window, "Generate");
    auto run = ui.request("trace");
    ui.peer.send(Object{{"id", run["id"]}, {"done", false}, {"event", trace_event(0, {1, 2}, 2)}});
    for (int t = 3; t <= 5; ++t)
        ui.peer.send(Object{{"id", run["id"]}, {"done", false}, {"event", trace_event(t - 1, {t}, 2)}});
    ui.peer.reply(run, Object{{"tokens", 3.0}, {"finish_reason", "length"}});
    for (int i = 0; i < 4; ++i) ui.frame();
    CHECK(ui.captured.find("Complete") != std::string::npos);           // the computation is done
    CHECK(ui.captured.find(" w3 w4 w5") == std::string::npos);          // the view is not
    // Back to full speed: everything shows at once.
    ImGui::FocusWindow(trace);
    ImGui::ActivateItemByID(trace->GetID("##trace.speed"));
    ui.frame();
    for (int i = 0; i < 14; ++i) {
        io.AddKeyEvent(ImGuiMod_Shift, true); io.AddKeyEvent(ImGuiKey_RightArrow, true); ui.frame();
        io.AddKeyEvent(ImGuiKey_RightArrow, false); io.AddKeyEvent(ImGuiMod_Shift, false); ui.frame();
    }
    io.AddKeyEvent(ImGuiKey_Enter, true); ui.frame(); io.AddKeyEvent(ImGuiKey_Enter, false); ui.frame();
    for (int i = 0; i < 3; ++i) ui.frame();
    CHECK(ui.captured.find("Full speed") != std::string::npos);
    CHECK(ui.captured.find(" w3 w4 w5") != std::string::npos);
    CHECK(ui.errors == 0);
}

TEST_CASE("the tape shows the whole trace and a click on it moves the grid's cursor there") {
    Ui ui;
    auto hello = ui.request("hello");
    ui.peer.reply(hello, Object{{"protocol", 1.0}, {"capabilities", Array{"generate", "list_tensors", "show_tensor", "trace"}}});
    auto list = ui.request("list_tensors");
    ui.peer.reply(list, Object{{"model", "Test model"}, {"tensors", Array{
        Object{{"name", "block.weight"}, {"group", "Block"}, {"shape", Array{4.0, 3.0}}, {"dtype", "float32"},
               {"bytes", 48.0}, {"size", 12.0}, {"rows", 4.0}, {"cols", 3.0}}}}});
    for (int i = 0; i < 3; ++i) ui.frame();
    ui.activate("Model 3D###caliper.inspector.model3d", "Trace");
    ui.activate(inference_window, "Generate");
    auto run = ui.request("trace");
    ui.peer.send(Object{{"id", run["id"]}, {"done", false}, {"event", trace_event(0, {1, 2, 3, 4, 5, 6}, 2)}});
    ui.peer.reply(run, Object{{"tokens", 4.0}, {"finish_reason", "length"}});
    for (int i = 0; i < 4; ++i) ui.frame();
    CHECK(ui.captured.find("Position 5") != std::string::npos);            // the cursor follows the newest token
    // Find the tape by its tooltip, near its left end, then click there.
    ImGuiWindow* trace = ImGui::FindWindowByName(trace_window);
    REQUIRE(trace != nullptr);
    auto& io = ImGui::GetIO();
    const float x = trace->Pos.x + 30;
    float found = -1;
    for (float y = trace->Pos.y; y < trace->Pos.y + trace->Size.y && found < 0; y += 2) {
        io.AddMousePosEvent(x, y);
        ui.frame(); ui.frame();
        if (ui.captured.find("move the grid here") != std::string::npos) found = y;
    }
    REQUIRE(found >= 0);
    CHECK(ui.captured.find("Position 0 (w1) of 6") != std::string::npos);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); ui.frame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); ui.frame(); ui.frame();
    CHECK(ui.captured.find("Position 0") != std::string::npos);
    CHECK(ui.captured.find("Position 5") == std::string::npos);
    CHECK(ui.errors == 0);
}
