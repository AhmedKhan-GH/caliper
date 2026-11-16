#pragma once
// ============================================================================
// widget_app.h: how the app gives the spec page's inline surface its agent
// (widget_surface.h, widget_agent_acp.h; design/widget-agent.md). The agent is
// caliper-agent over AcpClient when a provider is set up, and an agent that
// is never available otherwise, which leaves the prompt with only Ask in
// chat.
//
// What it needs:
//   - an interpreter with the sidecar's dependencies, CALIPER_AGENT_PYTHON;
//     until caliper-agent is packaged with a Python runtime there is no
//     other way to find it, and without it no agent is set up;
//   - the sidecar's package and the command table, from this build's source
//     tree (CALIPER_AGENT_SOURCE overrides the folder that holds
//     caliper_agent/ and commands/command_table.json);
//   - Caliper API running for this process, and an API file of the
//     sidecar's own that the app gives it, with a token of the origin agent
//     issued for it (AppApi::issue_agent_file()), which the sidecar reads to
//     reach the session's commands; without it no agent is set up;
//   - the provider's key, from the operating system's key store, placed
//     only in the sidecar's environment at spawn (place_key_in_environment(),
//     which also passes CALIPER_AGENT_OFFLINE through when the app has it and
//     removes the test switch).
//
// Tests and screenshots. A hidden app (--hidden) never touches the key
// store. When its environment holds CALIPER_AGENT_TEST_SCRIPT, which only a
// test sets, the sidecar inherits it with the rest of the app's environment
// and plays that script instead of a model, and the prompt row names the
// agent "Scripted"; a hidden app without it has no provider. A normal run
// ignores the switch.
//
// Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include "builder/widget_agent.h"

#include <memory>
#include <string>

namespace caliper_host {

struct AppWidgetEnv {
    bool hidden = false;
    std::string workspace;     // the session's folder: the sidecar's cwd
    std::string api_file;      // the sidecar's own API file (an agent token), or "" when the API is not running
    std::string python;        // CALIPER_AGENT_PYTHON
    std::string source;        // the folder with caliper_agent/ and commands/ (CALIPER_AGENT_SOURCE or the build's)
    std::string test_script;   // CALIPER_AGENT_TEST_SCRIPT, read only when hidden
};

// The environment of this process: the variables above, and the sidecar's
// API file, which the app issued for it ("" when the API is not running).
AppWidgetEnv app_widget_env(bool hidden, const std::string& workspace, const std::string& api_file = {});

// The agent for that environment; never null. `why`, when given, says why
// none is set up ("" when one is).
std::unique_ptr<WidgetAgent> make_app_widget_agent(const AppWidgetEnv& env, std::string* why = nullptr);

}  // namespace caliper_host
