// widget_app.cpp: the widget agent the app gives the spec page's inline
// surface (widget_app.h).
#include "builder/widget_app.h"

#include "builder/widget_agent_acp.h"

#include "app_paths.h"

#include <caliper_builder/agent/key_store.h>

#include <nlohmann/json.hpp>

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
    AppWidgetEnv e;
    e.hidden = hidden;
    e.workspace = workspace;
    e.python = env_text("CALIPER_AGENT_PYTHON");
    e.source = env_text("CALIPER_AGENT_SOURCE");
    if (e.source.empty()) e.source = CALIPER_WIDGET_AGENT_SOURCE;
    if (hidden) e.test_script = env_text("CALIPER_AGENT_TEST_SCRIPT");
    // The sidecar's own API file, with an agent token the app issued for it.
    e.api_file = api_file;
    return e;
}

std::unique_ptr<WidgetAgent> make_app_widget_agent(const AppWidgetEnv& env, std::string* why) {
    auto none = [why](const std::string& reason) -> std::unique_ptr<WidgetAgent> {
        if (why) *why = reason;
        return std::make_unique<NoWidgetAgent>(reason);
    };
    namespace fs = std::filesystem;
    std::error_code ec;
    if (env.python.empty()) return none("caliper-agent is not set up (CALIPER_AGENT_PYTHON)");
    const fs::path source(env.source);
    if (env.source.empty() || !fs::is_directory(source / "agent" / "caliper_agent", ec))
        return none("caliper-agent's package is not found");
    if (env.api_file.empty()) return none("Caliper API is not running (--api)");
    if (env.hidden && env.test_script.empty()) return none("no provider in a hidden app without a test script");

    AcpWidgetAgentOptions o;
    o.argv = {env.python, "-P", "-m", "caliper_agent", "--command-table",
              (source / "commands" / "command_table.json").string(), "--api-file", env.api_file};
    o.env = {{"PYTHONPATH", (source / "agent").string()}, {"PYTHONDONTWRITEBYTECODE", "1"}};
    o.cwd = env.workspace;
    if (env.hidden) {
        // A test: the sidecar inherits the script switch, the test's fake key
        // and CALIPER_AGENT_OFFLINE from this process's environment, and no
        // key store is touched.
        o.keys = nullptr;
        o.key_required = false;
        o.label = "Scripted";
    } else {
        static std::unique_ptr<caliper_builder::agent::KeyStore> keys = caliper_builder::agent::make_system_key_store();
        o.keys = keys.get();
        o.key_required = true;
        o.env_remove = {"CALIPER_AGENT_TEST_SCRIPT"};
    }
    if (why) why->clear();
    return std::make_unique<AcpWidgetAgent>(std::move(o));
}

}  // namespace caliper_host
