// caliper-agent-settings-shot: a development tool that draws the widget
// agent's settings (agent_settings.h) offscreen in a named state and writes a
// PNG of the canvas, as caliper-widget-shot does for the inline surface. It
// opens no window, is not installed, and never touches the system's key
// store: its keys live in an in-memory store, and every caliper-agent it
// starts runs with CALIPER_AGENT_OFFLINE=1 against a fake model server on
// loopback.
//
//   caliper-agent-settings-shot --scenario NAME --out FILE --data-dir DIR [options]
//
// Scenarios:
//   claude     the Claude preset with no key: Key "Not set"
//   key-set    the Claude preset after a fake key is saved in the in-memory
//              store: Key "Set", with Replace and Remove
//   local      the Ollama preset on a fake server: its listed models and the
//              probe's answer
//   no-tools   the Ollama preset on a fake server that refuses tools: the
//              probe finds no tool calling
//   gpu-note   as local, with a live run attached: the Shared GPU note
//   large-cache  as local, on a fake Ollama server whose model metadata gives
//              the cache a token takes, after the person raised the context
//              length to 262,144 tokens: the cache note
// Options:
//   --size WxH         the canvas (1440x900)
//   --python PATH      the interpreter with caliper-agent's dependencies
//                      (local, no-tools and gpu-note)
//   --agent-dir DIR    src/builder/agent, where caliper-agent and the fake
//                      server are (local, no-tools and gpu-note)
//   --frames N         frames drawn (4)
//
// It prints one JSON object: the file, the scenario, the settings' state and
// what the surface shows (never a key).
#include "builder/agent_settings.h"
#include "builder/tests/fake_model_server.h"
#include "spec_runtime/spec_offscreen.h"
#include "spec_runtime/spec_runtime.h"

#include "export_service.h"

#include <caliper_builder/agent/key_store.h>

#include <imgui.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;
using namespace caliper_host;

namespace {

int usage(const char* why) {
    std::fprintf(stderr, "caliper-agent-settings-shot: %s\n", why);
    std::fprintf(stderr,
                 "usage: caliper-agent-settings-shot --scenario claude|key-set|local|no-tools|gpu-note|large-cache --out FILE\n"
                 "                                   --data-dir DIR [--size WxH] [--python PATH] [--agent-dir DIR]\n"
                 "                                   [--frames N]\n");
    return 2;
}

int fail(const std::string& message) {
    std::fprintf(stderr, "caliper-agent-settings-shot: %s\n", message.c_str());
    return 1;
}

// Waits for the settings' probe to answer, polling as the app's frame does.
bool settle(AgentSettings& s, int timeout_s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (s.busy() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    s.frame();
    return !s.busy();
}

}  // namespace

int main(int argc, char** argv) {
    std::string scenario, out, data_dir, python, agent_dir;
    int width = 1440, height = 900, frames = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (!v) return usage(("missing a value after " + a).c_str());
        if (a == "--scenario") scenario = v;
        else if (a == "--out") out = v;
        else if (a == "--data-dir") data_dir = v;
        else if (a == "--python") python = v;
        else if (a == "--agent-dir") agent_dir = v;
        else if (a == "--frames") frames = std::atoi(v);
        else if (a == "--size") {
            if (std::sscanf(v, "%dx%d", &width, &height) != 2) return usage("--size takes WxH, such as 1440x900");
        } else return usage(("unknown option " + a).c_str());
        ++i;
    }
    const bool local = scenario == "local" || scenario == "no-tools" || scenario == "gpu-note" ||
                       scenario == "large-cache";
    if (scenario != "claude" && scenario != "key-set" && !local) return usage("unknown --scenario");
    if (out.empty() || data_dir.empty()) return usage("--out and --data-dir are required");
    if (local && (python.empty() || agent_dir.empty())) return usage("this scenario needs --python and --agent-dir");
    if (width < 64 || height < 64 || width > 8192 || height > 8192) return usage("--size must be from 64x64 to 8192x8192");
    if (frames < 1 || frames > 120) return usage("--frames must be from 1 to 120");
    std::error_code ec;
    fs::create_directories(data_dir, ec);

    // The keys of this tool live in memory only.
    caliper_builder::agent::MemoryKeyStore keys;
    std::unique_ptr<test::FakeModelServer> server;
    if (local) {
        std::string why;
        std::vector<std::string> args = {"--models", "local-small,local-large", "--context", "32768"};
        if (scenario == "no-tools") args = {"--models", "local-text-only", "--refuse-tools"};
        if (scenario == "large-cache") args = {"--models", "local-small,local-large", "--ollama-show"};
        server = test::FakeModelServer::start(python, agent_dir, args, &why);
        if (!server) return fail("the fake model server: " + why);
    }

    AgentSettings settings;
    AgentSettings::Options o;
    o.settings_path = (fs::path(data_dir) / "widget-agent.json").string();
    o.cache_dir = data_dir;
    o.keys = &keys;
    o.python = python;
    o.source = agent_dir.empty() ? std::string() : fs::path(agent_dir).parent_path().string();
    o.probe_env = {{"CALIPER_AGENT_OFFLINE", "1"}};
    const bool live = scenario == "gpu-note";
    o.live_attached = [live] { return live; };
    settings.start(o);
    settings.set_open(true);

    if (scenario == "claude" || scenario == "key-set") {
        settings.choose_preset("anthropic");
        settings.set_model("claude-haiku-5-5");
        settings.set_effort("low");
        if (scenario == "key-set") {
            settings.begin_key_entry();
            settings.set_key_text("sk-ant-api03-TESTONLY-settings-shot-0123456789abcdef");
            if (!settings.save_key()) return fail("the fake key was not saved: " + settings.key_problem());
        }
    } else {
        settings.choose_preset("ollama");
        settings.set_base_url(server->base_url());
        settings.list_models();
        if (!settle(settings, 120)) return fail("the model list did not come");
        settings.set_model(scenario == "no-tools" ? "local-text-only" : "local-large");
        settings.probe(true);
        if (!settle(settings, 180)) return fail("the probe did not answer");
        if (scenario == "large-cache") {
            // As the person typing 262144 into the Context length field.
            settings.set_override("context_length", "262144");
            if (!settle(settings, 120)) return fail("the override did not answer");
        }
    }

    std::string why;
    std::unique_ptr<SpecOffscreen> canvas = SpecOffscreen::create(width, height, 1.0f, &why);
    if (!canvas) return fail("cannot draw offscreen: " + why);
    std::vector<uint8_t> rgba;
    const bool ok = canvas->render(frames, [&](const ImRect& area) {
        settings.draw(area.Min.x, area.Min.y, area.GetWidth(), area.GetHeight(), canvas->look());
    }, &rgba, &why);
    if (!ok) return fail("the canvas: " + why);
    if (!export_write_png_atomic(out, rgba.data(), (uint32_t)width, (uint32_t)height, &why))
        return fail("cannot write " + out + ": " + why);

    json shown = json::array();
    for (const auto& t : settings.visible_text()) shown.push_back(t);
    json result{{"file", out},
                {"scenario", scenario},
                {"size", {width, height}},
                {"state", settings.state()},
                {"key", settings.key_status() == AgentSettings::KeyStatus::set ? "set" : "not set"},
                {"gpu_note", settings.gpu_note()},
                {"cache_note", settings.cache_note()},
                {"models", settings.model_choices()},
                {"shown", shown}};
    canvas->with_context([&] { settings.shutdown(); });
    canvas.reset();
    std::printf("%s\n", result.dump().c_str());
    return 0;
}
