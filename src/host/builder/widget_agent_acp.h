#pragma once
// ============================================================================
// widget_agent_acp.h: the widget agent over ACP (widget_agent.h,
// design/widget-agent.md, How it stays an ACP agent). AcpWidgetAgent drives
// the caliper-agent sidecar with an AcpClient:
//   - the sidecar starts as a child at the first start(), with the provider's
//     key placed only in its environment by the key store
//     (place_key_in_environment()), and stays for later sessions;
//   - start() opens a session with `session/new` and
//     _meta.caliper.widget = {target, preview, scope}, closing the one before with
//     `session/close`; the model and effort shown are the session's config
//     options as the sidecar reports them;
//   - prompt() is a `session/prompt` with the context bundle's blocks before
//     the request on a session's first prompt, whose updates become progress (each tool
//     call's title), thoughts and the reply, and whose answer's
//     _meta.caliper.widget gives the edits, the handoff and the limit;
//   - cancel() is `session/cancel`, close() `session/close`.
// The sidecar sends each edit to Caliper API with X-Caliper-Preview, so edits
// stack in the app's preview; it never commits.
//
// The blocking calls (the process start, session/new, session/close) run on
// a worker thread of the agent's own; their outcome and every ACP event
// reach the frame thread through poll(). Events of a session that has been
// closed are dropped.
//
// Built only with CALIPER_WITH_BUILDER.
// ============================================================================
#include "builder/widget_agent.h"

#include <caliper_builder/acp/subprocess.h>

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace caliper_builder::acp {
class AcpClient;
}
namespace caliper_builder::agent {
class KeyStore;
}

namespace caliper_host {

struct AcpWidgetAgentOptions {
    // The sidecar's command line, such as
    // {python, "-P", "-m", "caliper_agent", "--command-table", <table>, "--api-file", <api.json>}.
    std::vector<std::string> argv;
    std::vector<caliper_builder::acp::EnvVar> env;   // extra variables, never a key
    std::vector<std::string> env_remove;
    std::string cwd;                                 // the workspace folder: the process's and the session's
    // Where the provider's key lives; it reaches the sidecar only through its
    // environment. Null, or a provider that needs none (a local server),
    // starts it without a key.
    caliper_builder::agent::KeyStore* keys = nullptr;
    std::string provider = "anthropic";
    bool key_required = true;
    int request_timeout_ms = 60000;
    int cancel_timeout_ms = 10000;
    // Shown in the prompt row instead of the session's model and effort:
    // "Scripted" when the sidecar plays a test script, so that no capture
    // credits a model that did not run.
    std::string label;
};

class AcpWidgetAgent : public WidgetAgent {
public:
    explicit AcpWidgetAgent(AcpWidgetAgentOptions options);
    ~AcpWidgetAgent() override;   // closes the session and stops the sidecar
    AcpWidgetAgent(const AcpWidgetAgent&) = delete;
    AcpWidgetAgent& operator=(const AcpWidgetAgent&) = delete;

    bool available(std::string* why) const override;
    void start(const WidgetTarget& target, const std::string& preview, const std::vector<std::string>& scope) override;
    void prompt(const std::string& text, const nlohmann::json& context) override;
    void cancel() override;
    void close() override;
    bool poll(WidgetAgentEvent* out) override;
    std::string model_label() const override;
    std::vector<WidgetModelChoice> models() const override;
    void set_model(const std::string& value) override;
    // The model choices of session/new's configOptions.
    static std::vector<WidgetModelChoice> models_from_config(const nlohmann::json& config_options);

    // The session's _meta as start() sends it.
    static nlohmann::json session_meta(const WidgetTarget& target, const std::string& preview,
                                       const std::vector<std::string>& scope = {});
    // The prompt's content: the context blocks, then the request as text.
    static nlohmann::json prompt_content(const std::string& text, const nlohmann::json& context);
    // "Model name · effort" from session/new's configOptions, or "".
    static std::string label_from_config(const nlohmann::json& config_options);

    // Whether the sidecar process runs (for tests).
    bool running() const;

private:
    void post(std::function<void()> task);
    void push(WidgetAgentEvent e);
    void worker();

    AcpWidgetAgentOptions options_;
    std::shared_ptr<caliper_builder::acp::AcpClient> client_;   // made and replaced on the worker
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> tasks_;
    std::deque<WidgetAgentEvent> events_;
    std::string session_;    // the open session's id; events of others are dropped
    std::string label_;
    nlohmann::json config_;  // the session's config options
    std::string model_;      // the model the person chose, for this session and the later ones
    int generation_ = 0;     // grows at each start() and close()
    bool stop_ = false;
    std::thread thread_;
};

}  // namespace caliper_host
