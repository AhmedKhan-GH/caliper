// widget_app.cpp: the widget agent the app gives the spec page's inline
// surface (widget_app.h).
#include "builder/widget_app.h"

#include "builder/agent_settings.h"
#include "builder/widget_agent_acp.h"

#include "app_paths.h"

#include <caliper_builder/agent/key_store.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#ifndef CALIPER_WIDGET_AGENT_SOURCE
#define CALIPER_WIDGET_AGENT_SOURCE ""
#endif

namespace caliper_host {

namespace {

// No provider: the prompt offers only Ask in chat, and nothing starts.
class NoWidgetAgent final : public WidgetAgent {
public:
    explicit NoWidgetAgent(std::string why) : why_(std::move(why)) {}
    bool available(std::string* why) const override {
        if (why) *why = why_;
        return false;
    }
    void start(const WidgetTarget&, const std::string&, const std::vector<std::string>&) override {}
    void prompt(const std::string&, const nlohmann::json&) override {}
    void cancel() override {}
    void close() override {}
    bool poll(WidgetAgentEvent*) override { return false; }

private:
    std::string why_;
};

std::string env_text(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}


}  // namespace

AppWidgetEnv app_widget_env(bool hidden, const std::string& workspace, const std::string& api_file) {
    namespace fs = std::filesystem;
    AppWidgetEnv e;
    e.hidden = hidden;
    e.workspace = workspace;
    e.python = env_text("CALIPER_AGENT_PYTHON");
    e.source = env_text("CALIPER_AGENT_SOURCE");
    if (e.source.empty()) {
        const fs::path built(CALIPER_WIDGET_AGENT_SOURCE);
        std::error_code ec;
        if (!built.empty() && fs::is_directory(built / "agent" / "caliper_agent", ec)) e.source = built.string();
    }
    if (hidden) e.test_script = env_text("CALIPER_AGENT_TEST_SCRIPT");
    e.settings_path = caliper::app_data_path("widget-agent.json");
    e.cache_dir = caliper::app_data_dir();
    // The sidecar's own API file, with an agent token the app issued for it.
    e.api_file = api_file;
    return e;
}

bool app_widget_agent_options(const AppWidgetEnv& env, AcpWidgetAgentOptions* o, std::string* why) {
    auto none = [why](const std::string& reason) {
        if (why) *why = reason;
        return false;
    };
    namespace fs = std::filesystem;
    std::error_code ec;
    if (env.python.empty()) return none("caliper-agent is not set up (CALIPER_AGENT_PYTHON)");
    const bool has_source = !env.source.empty();
    const fs::path source(env.source);
    if (has_source && !fs::is_directory(source / "agent" / "caliper_agent", ec))
        return none("caliper-agent's package is not found");
    if (env.api_file.empty()) return none("Caliper API is not running (--api)");
    AgentSettingsData settings;
    if (!env.settings_path.empty()) read_agent_settings(env.settings_path, &settings);

    *o = AcpWidgetAgentOptions{};
    if (has_source) {
        o->argv = {env.python, "-P", "-m", "caliper_agent", "--command-table",
                   (source / "commands" / "command_table.json").string(), "--api-file", env.api_file};
        o->env = {{"PYTHONPATH", (source / "agent").string()}, {"PYTHONDONTWRITEBYTECODE", "1"}};
    } else {
        o->argv = {env.python, "-P", "-m", "caliper_agent", "--api-file", env.api_file};
        o->env = {{"PYTHONDONTWRITEBYTECODE", "1"}};
    }
    o->cwd = env.workspace;
    if (env.hidden && settings.preset.empty()) {
        if (env.test_script.empty()) return none("no provider in a hidden app without a test script");
        // A test: the sidecar inherits the script switch, the test's fake key
        // and CALIPER_AGENT_OFFLINE from this process's environment, and no
        // key store is touched.
        o->keys = nullptr;
        o->key_required = false;
        o->label = "Scripted";
        if (why) why->clear();
        return true;
    }
    // The chosen provider, from the settings, with the app's key store: the
    // system's for a normal run, an in-memory one for a hidden app.
    caliper_builder::agent::KeyStore* keys = app_key_store(env.hidden);
    const double now =
        (double)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const AgentLaunch launch = agent_launch(settings, keys, env.cache_dir, now);
    if (!launch.usable) return none(launch.why);
    o->argv.insert(o->argv.end(), launch.args.begin(), launch.args.end());
    o->keys = keys;
    o->provider = launch.account;
    o->key_required = launch.key_required;
    o->env_remove = {"CALIPER_AGENT_TEST_SCRIPT"};
    if (launch.local && env.live_attached) {
        std::function<bool()> live = env.live_attached;
        o->note = [live] { return live() ? std::string("Shared GPU") : std::string(); };
    }
    if (why) why->clear();
    return true;
}

std::unique_ptr<WidgetAgent> make_app_widget_agent(const AppWidgetEnv& env, std::string* why) {
    AcpWidgetAgentOptions o;
    std::string reason;
    if (!app_widget_agent_options(env, &o, &reason)) {
        if (why) *why = reason;
        return std::make_unique<NoWidgetAgent>(reason);
    }
    if (why) why->clear();
    return std::make_unique<AcpWidgetAgent>(std::move(o));
}

}  // namespace caliper_host
