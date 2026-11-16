// Tests of the widget agent's settings (agent_settings.h): the state for each
// preset, the key row's write-only behaviour through the KeyStore interface,
// the settings file, the sidecar's start-up options and environment, the
// remote-address warning, the GPU note, the probe's answer and the layout at
// two sizes (headless); the probe through caliper-agent against a fake model
// server on loopback (suite "agent"); and caliper-agent-settings-shot as a
// process (suite "gpu"). No test touches the system's key store, reaches a
// host off loopback or uses a real key.
#include "builder/agent_settings.h"
#include "builder/tests/fake_model_server.h"
#include "builder/widget_agent_acp.h"
#include "builder/widget_app.h"

#include <caliper_builder/acp/subprocess.h>
#include <caliper_builder/agent/key_store.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include <stb_image.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <thread>

using namespace caliper_host;
namespace agent = caliper_builder::agent;
namespace acp = caliper_builder::acp;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const std::string kKey = "sk-ant-api03-TESTONLY-settings-0123456789abcdefABCDEF";

std::string tmp_dir(const std::string& name) {
    const fs::path d = fs::path(CALIPER_SPEC_TEST_TMP) / "agent-settings" / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d.string();
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// A key store that counts what is asked of it, and holds keys in memory. Its
// do_read() is reached only through place_key_in_environment(), a friend of
// the interface; the settings must never reach it.
class SpyKeyStore final : public agent::KeyStore {
public:
    int sets = 0, has_calls = 0, removes = 0, reads = 0;
    std::map<std::string, std::string> keys;

protected:
    caliper_builder::Error do_set(const std::string& p, const std::string& k) override {
        ++sets;
        keys[p] = k;
        return {};
    }
    agent::KeyPresence do_has(const std::string& p) override {
        ++has_calls;
        return {{}, keys.count(p) != 0};
    }
    caliper_builder::Error do_remove(const std::string& p) override {
        ++removes;
        keys.erase(p);
        return {};
    }
    caliper_builder::Error do_read(const std::string& p, std::string* out) override {
        ++reads;
        auto it = keys.find(p);
        if (it == keys.end()) return {"keystore.not_found", "none"};
        *out = it->second;
        return {};
    }
};

AgentSettings::Options options_in(const std::string& dir, agent::KeyStore* keys) {
    AgentSettings::Options o;
    o.settings_path = (fs::path(dir) / "widget-agent.json").string();
    o.cache_dir = dir;
    o.keys = keys;
    o.now = [] { return 2.0e9; };
    return o;
}

bool contains(const std::vector<std::string>& v, const std::string& s) { return std::find(v.begin(), v.end(), s) != v.end(); }

void write_cache(const std::string& dir, const std::string& base_url, const std::string& model, json result,
                 json overrides = json::object(), double probed_at = 2.0e9 - 60) {
    json entry = {{"base_url", base_url}, {"model", model}, {"used_at", probed_at}, {"result", result},
                  {"overrides", overrides}};
    if (probed_at > 0) entry["probed_at"] = probed_at;
    std::ofstream(fs::path(dir) / "model-capabilities.json") << json{{"version", 1}, {"entries", {entry}}}.dump();
}

void set_env(const char* name, const char* value) {
    if (value) setenv(name, value, 1);
    else unsetenv(name);
}

const acp::EnvVar* find_var(const acp::SpawnOptions& s, const std::string& name) {
    for (const auto& v : s.env)
        if (v.name == name) return &v;
    return nullptr;
}

// Waits on the settings' state until no probe runs, then takes its answer. The
// deadline is sized for a slow runner; a probe still running after it fails
// the test here instead of letting it go on with no answer.
void settle(AgentSettings& s) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(150);
    while (s.busy() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE_MESSAGE(!s.busy(), "the probe did not answer within 150 s");
    s.frame();
}

// Sets variables in this process's environment and puts back what was there.
class EnvGuard {
public:
    void set(const std::string& name, const char* value) {
        if (!saved_.count(name)) {
            const char* old = std::getenv(name.c_str());
            saved_[name] = old ? std::optional<std::string>(old) : std::nullopt;
        }
        set_env(name.c_str(), value);
    }
    ~EnvGuard() {
        for (const auto& [name, value] : saved_) set_env(name.c_str(), value ? value->c_str() : nullptr);
    }

private:
    std::map<std::string, std::optional<std::string>> saved_;
};

// A store that has a key but cannot read it, as a locked keychain.
class LockedKeyStore final : public agent::KeyStore {
protected:
    caliper_builder::Error do_set(const std::string&, const std::string&) override { return {}; }
    agent::KeyPresence do_has(const std::string&) override { return {{}, true}; }
    caliper_builder::Error do_remove(const std::string&) override { return {}; }
    caliper_builder::Error do_read(const std::string&, std::string*) override {
        return {"keystore.failed", "The store is locked."};
    }
};

}  // namespace

TEST_SUITE_BEGIN("settings");

TEST_CASE("agent settings: the rows of each preset") {
    const std::string dir = tmp_dir("presets");
    agent::MemoryKeyStore keys;
    AgentSettings s;
    s.start(options_in(dir, &keys));
    CHECK(s.state() == "No provider");
    CHECK(s.key_status() == AgentSettings::KeyStatus::none);
    for (const auto& p : agent::provider_presets()) {
        CAPTURE(p.id);
        s.choose_preset(p.id);
        const AgentSettingsLayoutInput in = s.layout_input({0, 0, 1440, 900});
        CHECK(in.presets.size() == 7);
        CHECK(in.base_url_field == (p.local || p.id == "custom"));
        CHECK(in.key_row == (p.key != agent::KeyNeed::none && p.id != "custom"));   // Custom: after its address
        CHECK(in.efforts.size() == (p.id == "anthropic" ? 5u : 0u));
        CHECK(s.model_choices().size() == (p.id == "anthropic" ? 4u : 0u));
        CHECK(s.choice().base_url_editable == (p.local || p.id == "custom"));
        if (p.local) CHECK(s.choice().base_url == p.base_url.substr(0, p.base_url.size() - (p.base_url.back() == '/')));
    }
    s.choose_preset("custom");
    s.set_base_url("https://models.example.com/v1/");
    CHECK(s.choice().base_url == "https://models.example.com/v1");
    CHECK(s.key_status() == AgentSettings::KeyStatus::not_set);   // Custom's key, named by its address
    CHECK(s.choice().account == agent::custom_key_account("https://models.example.com/v1"));
    CHECK(s.state() == "No model");
    s.set_model("served-model");
    CHECK(s.state().empty());   // a Custom key is optional
    // Hosted presets keep their documented address.
    s.choose_preset("gemini");
    s.set_base_url("https://elsewhere.example.com/v1");
    CHECK(s.choice().base_url == "https://generativelanguage.googleapis.com/v1beta/openai");
    // The effort only where the preset takes one.
    s.set_effort("high");
    CHECK(s.data().current().effort.empty());
    s.choose_preset("anthropic");
    s.set_effort("medium");
    CHECK(s.choice().effort == "medium");
    s.set_effort("extreme");
    CHECK(s.choice().effort == "medium");
}

TEST_CASE("agent settings: the key row only ever writes") {
    const std::string dir = tmp_dir("key");
    SpyKeyStore keys;
    AgentSettings s;
    s.start(options_in(dir, &keys));
    s.choose_preset("anthropic");
    CHECK(s.key_status() == AgentSettings::KeyStatus::not_set);
    CHECK(s.state() == "No key");

    s.begin_key_entry();
    CHECK(s.key_editing());
    s.set_key_text(kKey);
    CHECK_FALSE(s.key_text_empty());
    CHECK(s.save_key());
    CHECK(s.key_text_empty());   // the field's text is wiped
    CHECK_FALSE(s.key_editing());
    CHECK(s.key_status() == AgentSettings::KeyStatus::set);
    CHECK(keys.keys.at("anthropic") == kKey);
    for (const std::string& t : s.visible_text()) CHECK(t.find(kKey) == std::string::npos);
    CHECK(contains(s.visible_text(), "Set"));
    CHECK(contains(s.visible_text(), "Replace"));
    CHECK(contains(s.visible_text(), "Remove"));
    const AgentSettingsLayoutInput in = s.layout_input({0, 0, 1440, 900});
    CHECK(in.key_set);
    CHECK_FALSE(in.key_field);

    // Replace, then Cancel: nothing written, the text wiped.
    s.begin_key_entry();
    s.set_key_text("sk-ant-api03-TESTONLY-other-0123456789");
    s.cancel_key();
    CHECK(s.key_text_empty());
    CHECK(keys.sets == 1);
    // A malformed key is refused and not kept.
    s.begin_key_entry();
    s.set_key_text("has a space 0123456789");
    CHECK_FALSE(s.save_key());
    CHECK(s.key_problem() == "Malformed key");
    CHECK(s.key_text_empty());
    // Remove.
    CHECK(s.remove_key());
    CHECK(s.key_status() == AgentSettings::KeyStatus::not_set);
    CHECK(keys.removes == 1);
    // One key per provider: OpenAI's is its own.
    s.choose_preset("openai");
    s.begin_key_entry();
    s.set_key_text("sk-TESTONLY-openai-0123456789");
    CHECK(s.save_key());
    CHECK(keys.keys.count("openai") == 1);
    CHECK(keys.keys.count("anthropic") == 0);
    // No local server has a key row, and its key cannot be entered.
    s.choose_preset("ollama");
    CHECK(s.key_status() == AgentSettings::KeyStatus::none);
    s.begin_key_entry();
    CHECK_FALSE(s.key_editing());
    // Nothing above asked the store for a key's text.
    CHECK(keys.reads == 0);
}

TEST_CASE("agent settings: widget-agent.json holds the choice and never a key") {
    const std::string dir = tmp_dir("file");
    agent::MemoryKeyStore keys;
    {
        AgentSettings s;
        s.start(options_in(dir, &keys));
        s.choose_preset("lmstudio");
        s.set_base_url("http://127.0.0.1:1234/v1");
        s.set_model("local-model");
        s.choose_preset("anthropic");
        s.set_model("claude-sonnet-5-5");
        s.begin_key_entry();
        s.set_key_text(kKey);
        REQUIRE(s.save_key());
    }
    const std::string text = read_text((fs::path(dir) / "widget-agent.json").string());
    CHECK(text.find(kKey) == std::string::npos);
    CHECK(text.find("sk-") == std::string::npos);
    AgentSettingsData d;
    REQUIRE(read_agent_settings((fs::path(dir) / "widget-agent.json").string(), &d));
    CHECK(d.preset == "anthropic");
    CHECK(d.providers.at("anthropic").model == "claude-sonnet-5-5");
    CHECK(d.providers.at("lmstudio").base_url == "http://127.0.0.1:1234/v1");
    CHECK(d.providers.at("lmstudio").model == "local-model");
    // A file that does not parse leaves the defaults; unknown presets are dropped.
    const std::string bad = (fs::path(dir) / "bad.json").string();
    std::ofstream(bad) << "{not json";
    std::string why;
    CHECK_FALSE(read_agent_settings(bad, &d, &why));
    CHECK(d.preset.empty());
    CHECK_FALSE(why.empty());
    std::ofstream(bad, std::ios::trunc) << R"({"version": 1, "preset": "nope", "providers": {"nope": {"model": "x"}}})";
    CHECK(read_agent_settings(bad, &d));
    CHECK(d.preset.empty());
    CHECK(d.providers.empty());
    // A missing file is the defaults.
    CHECK(read_agent_settings((fs::path(dir) / "missing.json").string(), &d));
}

TEST_CASE("agent settings: the sidecar's start-up options") {
    const std::string dir = tmp_dir("launch");
    agent::MemoryKeyStore keys;
    const double now = 2.0e9;
    AgentSettingsData d;
    CHECK(agent_launch(d, &keys, dir, now).why == "No provider");

    d.preset = "anthropic";
    CHECK(agent_launch(d, &keys, dir, now).why == "No key");
    REQUIRE(keys.set("anthropic", kKey).ok());
    d.providers["anthropic"].effort = "low";
    AgentLaunch L = agent_launch(d, &keys, dir, now);
    CHECK(L.usable);
    CHECK(L.args == std::vector<std::string>{"--preset", "anthropic", "--effort", "low", "--cache-dir", dir});
    CHECK(L.account == "anthropic");
    CHECK(L.key_required);
    CHECK_FALSE(L.local);

    d.preset = "ollama";
    CHECK(agent_launch(d, &keys, dir, now).why == "No model");
    d.providers["ollama"].model = "local-model";
    L = agent_launch(d, &keys, dir, now);
    CHECK(L.usable);
    CHECK(L.args == std::vector<std::string>{"--preset", "ollama", "--base-url", "http://localhost:11434/v1", "--model",
                                             "local-model", "--cache-dir", dir});
    CHECK(L.account == "ollama");
    CHECK_FALSE(L.key_required);
    CHECK(L.local);

    d.preset = "custom";
    d.providers["custom"].model = "m";
    CHECK(agent_launch(d, &keys, dir, now).why == "Invalid address");
    d.providers["custom"].base_url = "http://user:pw@127.0.0.1/v1";
    CHECK(agent_launch(d, &keys, dir, now).why == "Invalid address");
    d.providers["custom"].base_url = "https://models.example.com/v1";
    L = agent_launch(d, &keys, dir, now);
    CHECK(L.usable);
    CHECK(L.account == agent::custom_key_account("https://models.example.com/v1"));

    // A model the probe found without tool calling cannot drive Caliper.
    d.preset = "ollama";
    write_cache(dir, "http://localhost:11434/v1", "local-model", {{"tools", {{"value", false}, {"source", "probe"}}}});
    CHECK(agent_launch(d, &keys, dir, now).why == "No tool calling");
    // A stale answer is not relied on; the person's override is, without one.
    write_cache(dir, "http://localhost:11434/v1", "local-model", {{"tools", {{"value", false}, {"source", "probe"}}}},
                json::object(), now - 40.0 * 24 * 3600);
    CHECK(agent_launch(d, &keys, dir, now).usable);
    write_cache(dir, "http://localhost:11434/v1", "local-model", json::object(), {{"tools", false}}, -1);
    CHECK(agent_launch(d, &keys, dir, now).why == "No tool calling");
    write_cache(dir, "http://localhost:11434/v1", "local-model", {{"tools", {{"value", false}, {"source", "probe"}}}},
                {{"tools", true}});
    CHECK(agent_launch(d, &keys, dir, now).usable);
}

TEST_CASE("agent settings: the app's agent from the settings, and its environment") {
    const std::string dir = tmp_dir("app");
    agent::KeyStore* keys = app_key_store(true);   // a hidden app's: in memory
    REQUIRE(keys != nullptr);
    CHECK(dynamic_cast<agent::MemoryKeyStore*>(keys) != nullptr);   // never the system's store
    CHECK(keys == app_key_store(true));
    AppWidgetEnv env;
    env.hidden = true;
    env.workspace = dir;
    env.python = "python3";
    env.source = CALIPER_BUILDER_SOURCE_DIR;
    env.api_file = (fs::path(dir) / "api.json").string();
    env.settings_path = (fs::path(dir) / "widget-agent.json").string();
    env.cache_dir = dir;
    bool live = false;
    env.live_attached = [&live] { return live; };
    AcpWidgetAgentOptions o;
    std::string why;

    // No settings and no test script: no provider in a hidden app.
    CHECK_FALSE(app_widget_agent_options(env, &o, &why));
    CHECK_FALSE(why.empty());

    // Ollama on loopback: no key, the GPU note while a run is live.
    AgentSettingsData d;
    d.preset = "ollama";
    d.providers["ollama"] = {"http://127.0.0.1:11434/v1", "local-model", ""};
    REQUIRE(write_agent_settings(env.settings_path, d));
    REQUIRE(app_widget_agent_options(env, &o, &why));
    CHECK(o.keys == keys);
    CHECK(o.provider == "ollama");
    CHECK_FALSE(o.key_required);
    CHECK(std::find(o.argv.begin(), o.argv.end(), "--preset") != o.argv.end());
    CHECK(std::find(o.argv.begin(), o.argv.end(), "local-model") != o.argv.end());
    CHECK(std::find(o.env_remove.begin(), o.env_remove.end(), "CALIPER_AGENT_TEST_SCRIPT") != o.env_remove.end());
    REQUIRE(o.note);
    CHECK(o.note().empty());
    live = true;
    CHECK(o.note() == "Shared GPU");
    {
        AcpWidgetAgent a(o);   // made, not started
        CHECK(a.model_label() == "Shared GPU");
        CHECK(a.available(nullptr));
    }
    live = false;
    acp::SpawnOptions spawn;
    spawn.env = {{"ANTHROPIC_API_KEY", "inherited-0123456789"}, {"CALIPER_MODEL_API_KEY", "inherited-0123456789"}};
    set_env("CALIPER_AGENT_OFFLINE", "1");
    REQUIRE(agent::place_key_in_environment(*o.keys, o.provider, &spawn).ok());
    CHECK(find_var(spawn, "ANTHROPIC_API_KEY") == nullptr);
    CHECK(find_var(spawn, "CALIPER_MODEL_API_KEY") == nullptr);
    REQUIRE(find_var(spawn, "CALIPER_AGENT_OFFLINE") != nullptr);
    CHECK(find_var(spawn, "CALIPER_AGENT_OFFLINE")->value == "1");

    // The Claude preset: the key only in ANTHROPIC_API_KEY, never in argv.
    d.preset = "anthropic";
    REQUIRE(write_agent_settings(env.settings_path, d));
    CHECK_FALSE(app_widget_agent_options(env, &o, &why));
    CHECK(why == "No key");
    REQUIRE(keys->set("anthropic", kKey).ok());
    REQUIRE(app_widget_agent_options(env, &o, &why));
    CHECK(o.key_required);
    CHECK_FALSE(o.note);   // not a local server
    for (const auto& a : o.argv) CHECK(a.find(kKey) == std::string::npos);
    acp::SpawnOptions claude;
    REQUIRE(agent::place_key_in_environment(*o.keys, o.provider, &claude).ok());
    REQUIRE(find_var(claude, "ANTHROPIC_API_KEY") != nullptr);
    CHECK(find_var(claude, "ANTHROPIC_API_KEY")->value == kKey);
    CHECK(find_var(claude, "CALIPER_MODEL_API_KEY") == nullptr);
    CHECK(std::count(claude.env_remove.begin(), claude.env_remove.end(), std::string("CALIPER_MODEL_API_KEY")) == 1);
    REQUIRE(keys->remove("anthropic").ok());
    set_env("CALIPER_AGENT_OFFLINE", "1");
}

TEST_CASE("agent settings: the address's warnings") {
    AgentSettingsData d;
    d.preset = "ollama";
    d.providers["ollama"].base_url = "http://192.0.2.10:11434/v1";
    CHECK(agent_base_url_warnings(agent_choice(d)) == std::vector<std::string>{"Remote address", "Unencrypted"});
    d.providers["ollama"].base_url = "https://gpu.example.com/v1";
    CHECK(agent_base_url_warnings(agent_choice(d)) == std::vector<std::string>{"Remote address"});
    d.providers["ollama"].base_url = "http://127.0.0.1:11434/v1";
    CHECK(agent_base_url_warnings(agent_choice(d)).empty());
    d.providers["ollama"].base_url = "http://[::1]:11434/v1";
    CHECK(agent_base_url_warnings(agent_choice(d)).empty());
    d.providers["ollama"].base_url = "ftp://127.0.0.1/v1";
    CHECK(agent_base_url_warnings(agent_choice(d)) == std::vector<std::string>{"Invalid address"});
    d.preset = "custom";
    d.providers["custom"].base_url = "https://models.example.com/v1";
    CHECK(agent_base_url_warnings(agent_choice(d)) == std::vector<std::string>{"Remote address"});
    d.preset = "openai";   // a hosted preset: data leaves by design, no warning
    CHECK(agent_base_url_warnings(agent_choice(d)).empty());
    for (const char* bad : {"http://h/v1?key=1", "http://h/v1#x", "http://h:99999/v1", "http://h /v1", "h/v1", ""})
        CHECK_FALSE(agent_base_url_valid(bad));
    CHECK(agent_url_is_loopback("http://localhost:1234/v1"));
    CHECK(agent_url_is_loopback("http://127.0.0.1/v1"));
    CHECK_FALSE(agent_url_is_loopback("http://127.example.com/v1"));
    CHECK_FALSE(agent_url_is_loopback("http://10.0.0.1/v1"));
}

TEST_CASE("agent settings: the GPU note's condition") {
    AgentSettingsData d;
    CHECK(agent_gpu_note(true, agent_choice(d)).empty());
    d.preset = "ollama";
    CHECK(agent_gpu_note(true, agent_choice(d)) == "Shared GPU");
    CHECK(agent_gpu_note(false, agent_choice(d)).empty());
    d.providers["ollama"].base_url = "http://192.0.2.10:11434/v1";   // another machine's GPU
    CHECK(agent_gpu_note(true, agent_choice(d)).empty());
    d.preset = "custom";
    d.providers["custom"].base_url = "http://127.0.0.1:8080/v1";
    CHECK(agent_gpu_note(true, agent_choice(d)) == "Shared GPU");
    d.preset = "anthropic";
    CHECK(agent_gpu_note(true, agent_choice(d)).empty());
    // Through the component, with its live switch.
    const std::string dir = tmp_dir("gpu");
    agent::MemoryKeyStore keys;
    bool live = true;
    AgentSettings s;
    AgentSettings::Options o = options_in(dir, &keys);
    o.live_attached = [&live] { return live; };
    s.start(o);
    s.choose_preset("lmstudio");
    CHECK(s.gpu_note() == "Shared GPU");
    CHECK(s.layout_input({0, 0, 1440, 900}).note);
    live = false;
    CHECK(s.gpu_note().empty());
}

TEST_CASE("agent settings: the probe's answer and the cache") {
    const AgentProbeAnswer a = parse_probe_answer(R"({"version": 1, "models": ["a", "b"],
        "capabilities": {"tools": {"value": true, "source": "probe"}, "vision": {"value": false, "source": "override"},
                         "context_length": {"value": 32768, "source": "server"}},
        "overrides": {"vision": false}, "errors": [{"capability": "vision", "code": "probe.cut_off", "message": "x"}],
        "warnings": ["w"], "can_drive_caliper": true})");
    CHECK(a.ran);
    CHECK(a.models == std::vector<std::string>{"a", "b"});
    CHECK(a.caps.tools == std::optional<bool>(true));
    CHECK(a.caps.vision == std::optional<bool>(false));
    CHECK(a.caps.vision_source == "override");
    CHECK(a.caps.context == std::optional<long long>(32768));
    CHECK(a.problems == std::vector<std::string>{"Reply cut off"});
    CHECK(parse_probe_answer("not json").failure == "No answer");
    CHECK(probe_problem_name("probe.unreachable") == "Server unreachable");
    CHECK(probe_problem_name("agent.offline") == "Offline");

    const std::string dir = tmp_dir("cache");
    write_cache(dir, "http://127.0.0.1:1/v1", "m",
                {{"tools", {{"value", true}, {"source", "probe"}}}, {"context_length", {{"value", 4096}, {"source", "server"}}}},
                {{"vision", false}});
    const AgentCapabilities c = read_cached_capabilities(dir, "http://127.0.0.1:1/v1/", "m", 2.0e9);
    CHECK(c.tools == std::optional<bool>(true));
    CHECK(c.vision == std::optional<bool>(false));
    CHECK(c.vision_source == "override");
    CHECK(c.context == std::optional<long long>(4096));
    CHECK_FALSE(read_cached_capabilities(dir, "http://127.0.0.1:1/v1", "other", 2.0e9).tools.has_value());
    std::ofstream(fs::path(dir) / "model-capabilities.json", std::ios::trunc) << "garbage";
    CHECK_FALSE(read_cached_capabilities(dir, "http://127.0.0.1:1/v1", "m", 2.0e9).tools.has_value());
}

TEST_CASE("agent settings: the layout at two sizes") {
    const std::string dir = tmp_dir("layout");
    agent::MemoryKeyStore keys;
    AgentSettings s;
    AgentSettings::Options o = options_in(dir, &keys);
    o.live_attached = [] { return true; };
    s.start(o);
    s.choose_preset("lmstudio");
    s.set_base_url("http://192.0.2.10:1234/v1");   // two warnings
    const WidgetMeasure measure = [](const std::string& t, WidgetFont) { return 8.0f * (float)t.size(); };
    for (const auto& size : std::vector<std::pair<float, float>>{{1440, 900}, {2560, 1440}, {900, 640}}) {
        const float w = size.first, h = size.second;
        CAPTURE(w);
        const AgentSettingsLayoutInput in = s.layout_input({0, 30, w, h - 60});
        const AgentSettingsLayout L = layout_agent_settings(in, AgentSettingsMetrics::for_text_scale(1.0f), measure);
        const WidgetRect area{0, 30, w, h - 60};
        CHECK(area.contains(L.window));
        CHECK(L.window.w >= std::min(560.0f, area.w - 48));
        if (w >= 1440) CHECK(L.window.w == doctest::Approx(std::round(w * 0.62f)).epsilon(0.01));
        CHECK(L.two_columns == (w >= 1440));
        std::vector<WidgetRect> rects = L.presets;
        for (const auto& r : {L.base_url, L.model, L.list, L.tools, L.vision, L.context, L.probe})
            rects.push_back(r);
        for (const auto& r : L.warnings) rects.push_back(r);
        for (const auto& r : L.tools_buttons) rects.push_back(r);
        for (const auto& r : rects) CHECK(L.window.contains(r));
        // Fields and buttons never overlap one another (a row and its own
        // buttons are drawn over each other on purpose, so rows are left out).
        std::vector<WidgetRect> solid = L.presets;
        for (const auto& r : {L.base_url, L.model, L.list, L.probe}) solid.push_back(r);
        for (const auto& r : L.tools_buttons) solid.push_back(r);
        for (const auto& r : L.vision_buttons) solid.push_back(r);
        for (size_t i = 0; i < solid.size(); ++i)
            for (size_t j = i + 1; j < solid.size(); ++j) CHECK_FALSE(solid[i].overlaps(solid[j]));
        // The capabilities sit beside the model in two columns, under it in one.
        if (L.two_columns) CHECK(L.caps.x >= L.model.right());
        else CHECK(L.caps.y >= L.model.bottom());
        CHECK(L.warnings.size() == 2);
        CHECK(L.note.empty());   // another machine's GPU: no note
        // One placement: the top at 12 % of the work area's height, whatever
        // the content, while it fits.
        AgentSettingsLayoutInput fewer = in;
        fewer.warnings = 0;
        fewer.efforts.clear();
        const AgentSettingsLayout F = layout_agent_settings(fewer, AgentSettingsMetrics::for_text_scale(1.0f), measure);
        if (w >= 1440) {
            CHECK(L.window.y == doctest::Approx(30 + std::round((h - 60) * 0.12f)));
            CHECK(F.window.y == doctest::Approx(L.window.y));
            CHECK(F.window.h <= L.window.h);
        }
        // On this machine, with a live run: the note under the presets.
        AgentSettingsLayoutInput local = in;
        local.note = true;
        local.warnings = 0;
        const AgentSettingsLayout N = layout_agent_settings(local, AgentSettingsMetrics::for_text_scale(1.0f), measure);
        CHECK(N.window.contains(N.note));
        CHECK(N.note.y >= N.presets.back().bottom());
        CHECK(N.base_url.y >= N.note.bottom());
    }
}

TEST_SUITE_END();

// ---- the probe through caliper-agent, against a fake model server on loopback ----------

TEST_SUITE_BEGIN("agent");

TEST_CASE("agent settings: List and Probe through caliper-agent on a fake local server") {
    const std::string python = CALIPER_BUILDER_TEST_PYTHON;
    if (python.empty()) {
        MESSAGE("caliper_agent_settings_agent skipped: no CALIPER_BUILDER_TEST_PYTHON");
        return;
    }
    const std::string agent_dir = std::string(CALIPER_BUILDER_SOURCE_DIR) + "/agent";
    std::string why;
    auto server = test::FakeModelServer::start(python, agent_dir, {"--models", "local-a,local-b", "--context", "32768"},
                                               &why);
    REQUIRE_MESSAGE(server, why);
    const std::string dir = tmp_dir("probe");
    agent::MemoryKeyStore keys;
    AgentSettings s;
    AgentSettings::Options o = options_in(dir, &keys);
    o.now = {};
    o.python = python;
    o.source = CALIPER_BUILDER_SOURCE_DIR;
    o.probe_env = {{"CALIPER_AGENT_OFFLINE", "1"}};
    s.start(o);
    s.choose_preset("ollama");
    s.set_base_url(server->base_url());
    s.list_models();
    settle(s);
    CHECK(s.model_choices() == std::vector<std::string>{"local-a", "local-b"});
    s.set_model("local-b");
    s.probe(true);
    settle(s);
    AgentCapabilities c = s.capabilities();
    CHECK(c.tools == std::optional<bool>(true));
    CHECK(c.tools_source == "probe");
    CHECK(c.vision == std::optional<bool>(true));
    CHECK(c.context == std::optional<long long>(32768));
    CHECK(s.state().empty());
    // The person's override, kept in the probe's cache, which the app's agent reads too.
    s.set_override("tools", "no");
    settle(s);
    CHECK(s.state() == "No tool calling");
    const double now =
        (double)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    CHECK(agent_launch(s.data(), &keys, dir, now).why == "No tool calling");
    s.set_override("tools", "auto");
    settle(s);
    CHECK(s.state().empty());
    // A server that refuses tools.
    auto no_tools = test::FakeModelServer::start(python, agent_dir, {"--models", "text-only", "--refuse-tools"}, &why);
    REQUIRE_MESSAGE(no_tools, why);
    s.set_base_url(no_tools->base_url());
    s.set_model("text-only");
    s.probe(true);
    settle(s);
    CHECK(s.capabilities().tools == std::optional<bool>(false));
    CHECK(s.state() == "No tool calling");
    const std::string cache = read_text((fs::path(dir) / "model-capabilities.json").string());
    CHECK(cache.find("sk-") == std::string::npos);
}

TEST_CASE("agent settings: the probe never passes on a key or an endpoint from the app's own environment") {
    const std::string python = CALIPER_BUILDER_TEST_PYTHON;
    if (python.empty()) {
        MESSAGE("caliper_agent_settings_agent skipped: no CALIPER_BUILDER_TEST_PYTHON");
        return;
    }
    const std::string agent_dir = std::string(CALIPER_BUILDER_SOURCE_DIR) + "/agent";
    const std::string dir = tmp_dir("probe-env");
    const std::string record = dir + "/requests.jsonl";
    std::string why;
    auto server = test::FakeModelServer::start(python, agent_dir, {"--models", "local-a", "--record", record}, &why);
    REQUIRE_MESSAGE(server, why);
    // The app's own environment holds keys and endpoints of its own; none may
    // reach the server the person chose.
    const std::string inherited = "sk-TESTONLY-inherited-0123456789abcdef";
    EnvGuard env;
    for (const char* name : {"CALIPER_MODEL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"}) env.set(name, inherited.c_str());
    env.set("OPENAI_BASE_URL", "http://127.0.0.1:9/v1");
    env.set("ANTHROPIC_BASE_URL", "http://127.0.0.1:9");
    env.set("CALIPER_AGENT_OFFLINE", "1");
    auto requests = [&] {
        std::vector<json> out;
        std::istringstream lines(read_text(record));
        for (std::string line; std::getline(lines, line);)
            if (!line.empty()) out.push_back(json::parse(line));
        return out;
    };
    auto options = [&](const std::string& name, agent::KeyStore* keys) {
        AgentSettings::Options o = options_in(tmp_dir("probe-env-" + name), keys);
        o.now = {};
        o.python = python;
        o.source = CALIPER_BUILDER_SOURCE_DIR;
        o.probe_env = {{"CALIPER_AGENT_OFFLINE", "1"}};
        return o;
    };

    // A store that holds the Custom entry's key but cannot read it: the probe
    // does not run, as the sidecar's start would not, and nothing reaches the
    // server.
    {
        LockedKeyStore locked;
        AgentSettings s;
        s.start(options("locked", &locked));
        s.choose_preset("custom");
        s.set_base_url(server->base_url());
        s.set_model("local-a");
        s.probe(true);
        settle(s);
        CHECK(s.probe_answer().failure == "Unreadable key");
        CHECK_FALSE(s.probe_answer().ran);
        CHECK(requests().empty());
    }
    // No store at all, and a Custom entry, whose key is optional: the server
    // is still listed and probed, with no key from the environment.
    {
        AgentSettings s;
        s.start(options("no-store", nullptr));
        s.choose_preset("custom");
        s.set_base_url(server->base_url());
        s.list_models();
        settle(s);
        CHECK(s.model_choices() == std::vector<std::string>{"local-a"});
        s.set_model("local-a");
        s.probe(true);
        settle(s);
        CHECK(s.probe_answer().ran);
        CHECK(s.capabilities().tools == std::optional<bool>(true));
    }
    // A local server through the store, which needs no key: probed, no key.
    {
        agent::MemoryKeyStore keys;
        AgentSettings s;
        s.start(options("local", &keys));
        s.choose_preset("lmstudio");
        s.set_base_url(server->base_url());
        s.set_model("local-a");
        s.probe(true);
        settle(s);
        CHECK(s.probe_answer().ran);
    }
    const std::vector<json> seen = requests();
    CHECK(seen.size() >= 3);   // the list, then the probes' model list and test calls
    for (const json& r : seen) {
        CAPTURE(r.dump());
        const std::string auth = r["authorization"].is_string() ? r["authorization"].get<std::string>() : std::string();
        CHECK(auth.find(inherited) == std::string::npos);
        CHECK(auth.find("TESTONLY") == std::string::npos);
    }
    // A provider that requires a key, with none stored: refused at once,
    // before any child starts, as the sidecar's start refuses.
    {
        agent::MemoryKeyStore keys;
        AgentSettings s;
        s.start(options("required", &keys));
        s.choose_preset("openai");
        s.set_model("gpt-test");
        s.probe(true);
        CHECK_FALSE(s.busy());
        CHECK(s.probe_answer().failure == "No key");
        AgentSettings none;
        none.start(options("required-no-store", nullptr));
        none.choose_preset("gemini");
        none.set_model("gemini-test");
        none.probe(true);
        CHECK_FALSE(none.busy());
        CHECK(none.probe_answer().failure == "No key");
    }
}

TEST_SUITE_END();

// ---- caliper-agent-settings-shot as a process --------------------------------------------

TEST_SUITE_BEGIN("gpu");

TEST_CASE("agent settings: caliper-agent-settings-shot draws the Claude preset and a saved key") {
    const std::string dir = tmp_dir("shot");
    for (const std::string scenario : {"claude", "key-set"}) {
        CAPTURE(scenario);
        acp::SpawnOptions o;
        const std::string file = (fs::path(dir) / (scenario + ".png")).string();
        o.argv = {CALIPER_AGENT_SETTINGS_SHOT_EXE, "--scenario", scenario, "--out", file, "--data-dir",
                  (fs::path(dir) / scenario).string(), "--size", "960x600"};
        o.env = {{"CALIPER_AGENT_OFFLINE", "1"}};
        acp::RunLimits limits;
        limits.timeout_ms = 120000;
        const acp::RunResult run = acp::run_process(o, limits);
        if (run.out.empty() && run.err.find("cannot draw offscreen") != std::string::npos) {
            MESSAGE("caliper_agent_settings_gpu skipped: no offscreen canvas");
            return;
        }
        REQUIRE_MESSAGE(run.status.success(), run.err);
        const json j = json::parse(run.out);
        CHECK(j["key"] == (scenario == "key-set" ? "set" : "not set"));
        CHECK(run.out.find("TESTONLY") == std::string::npos);
        int w = 0, h = 0, n = 0;
        unsigned char* px = stbi_load(file.c_str(), &w, &h, &n, 4);
        REQUIRE(px != nullptr);
        CHECK(w == 960);
        CHECK(h == 600);
        stbi_image_free(px);
    }
}

TEST_SUITE_END();
