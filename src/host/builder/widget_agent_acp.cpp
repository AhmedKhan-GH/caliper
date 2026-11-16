// widget_agent_acp.cpp: the widget agent over ACP and the caliper-agent
// sidecar (widget_agent_acp.h).
#include "builder/widget_agent_acp.h"

#include <caliper_builder/acp/acp_client.h>
#include <caliper_builder/agent/key_store.h>

#include <chrono>
#include <utility>

namespace caliper_host {

using nlohmann::json;
namespace acp = caliper_builder::acp;

AcpWidgetAgent::AcpWidgetAgent(AcpWidgetAgentOptions options) : options_(std::move(options)) {
    thread_ = std::thread([this] { worker(); });
}

AcpWidgetAgent::~AcpWidgetAgent() {
    close();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    if (client_) client_->shutdown();
}

void AcpWidgetAgent::post(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push_back(std::move(task));
    }
    cv_.notify_all();
}

void AcpWidgetAgent::push(WidgetAgentEvent e) {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back(std::move(e));
}

void AcpWidgetAgent::worker() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (tasks_.empty()) return;   // stop_, with nothing left
            task = std::move(tasks_.front());
            tasks_.pop_front();
        }
        task();
    }
}

bool AcpWidgetAgent::available(std::string* why) const {
    if (!options_.key_required) return true;
    if (!options_.keys) {
        if (why) *why = "no key store";
        return false;
    }
    const caliper_builder::agent::KeyPresence p = options_.keys->has(options_.provider);
    if (!p.present && why) *why = p.error ? p.error.message : "no key";
    return p.present;
}

bool AcpWidgetAgent::running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return client_ && client_->agent_running();
}

std::string AcpWidgetAgent::model_label() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return label_;
}

json AcpWidgetAgent::session_meta(const WidgetTarget& target, const std::string& preview,
                                  const std::vector<std::string>& scope) {
    json w{{"target", target.to_json()}, {"preview", preview}};
    if (!scope.empty()) w["scope"] = scope;
    return {{"caliper", {{"widget", w}}}};
}

json AcpWidgetAgent::prompt_content(const std::string& text, const json& context) {
    json blocks = context.is_array() ? context : json::array();
    blocks.push_back({{"type", "text"}, {"text", text}});
    return blocks;
}

std::string AcpWidgetAgent::label_from_config(const json& config) {
    if (!config.is_array()) return {};
    std::string model, effort;
    for (const json& o : config) {
        if (!o.is_object() || !o.contains("currentValue") || !o["currentValue"].is_string()) continue;
        const std::string id = o.value("id", ""), value = o["currentValue"].get<std::string>();
        if (id == "model") {
            model = value;
            // The option's name, such as "Claude Haiku 5.5", over its value.
            for (const json& v : o.value("options", json::array()))
                if (v.is_object() && v.value("value", "") == value && v.contains("name") && v["name"].is_string())
                    model = v["name"].get<std::string>();
        } else if (id == "effort") {
            effort = value;
        }
    }
    if (model.empty()) return {};
    return effort.empty() ? model : model + " \xC2\xB7 " + effort;   // ·
}

void AcpWidgetAgent::start(const WidgetTarget& target, const std::string& preview, const std::vector<std::string>& scope) {
    int gen = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        gen = ++generation_;
        session_.clear();
    }
    const json meta = session_meta(target, preview, scope);
    post([this, gen, meta] {
        auto stale = [&] {
            std::lock_guard<std::mutex> lock(mutex_);
            return gen != generation_ || stop_;
        };
        auto failed = [&](caliper_builder::Error error) {
            if (stale()) return;
            WidgetAgentEvent e;
            e.kind = WidgetAgentEvent::Kind::failed;
            e.error = std::move(error);
            push(std::move(e));
        };
        // The sidecar: started once, again after it stopped.
        std::shared_ptr<acp::AcpClient> client;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            client = client_;
        }
        if (!client || !client->agent_running()) {
            if (client) client->shutdown();
            acp::SpawnOptions spawn;
            spawn.argv = options_.argv;
            spawn.env = options_.env;
            spawn.env_remove = options_.env_remove;
            if (options_.key_required) {
                if (!options_.keys) return failed({"keystore.not_found", "No key store is set up.", ""});
                // The key reaches the child only through its environment.
                if (caliper_builder::Error e = caliper_builder::agent::place_key_in_environment(
                        *options_.keys, options_.provider, &spawn))
                    return failed(e);
            }
            acp::AcpClientOptions o;
            o.agent_argv = spawn.argv;
            o.agent_env = std::move(spawn.env);
            o.agent_env_remove = spawn.env_remove;
            o.agent_cwd = options_.cwd;
            o.request_timeout_ms = options_.request_timeout_ms;
            o.cancel_timeout_ms = options_.cancel_timeout_ms;
            client = std::make_shared<acp::AcpClient>(std::move(o));
            {
                std::lock_guard<std::mutex> lock(mutex_);
                client_ = client;
            }
            if (caliper_builder::Error e = client->start()) return failed(e);
        }
        // One session at a time: one still open is closed first.
        if (!client->session_id().empty() && !client->turn_active()) (void)client->close_session();
        acp::SessionInfo info;
        if (caliper_builder::Error e = client->new_session(options_.cwd, {}, &info, meta)) return failed(e);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (gen != generation_ || stop_) return;   // closed meanwhile: close() follows in the queue
            session_ = info.id;
            label_ = label_from_config(info.config_options);
        }
        WidgetAgentEvent e;
        e.kind = WidgetAgentEvent::Kind::started;
        e.model = label_from_config(info.config_options);
        push(std::move(e));
    });
}

void AcpWidgetAgent::prompt(const std::string& text, const json& context) {
    int gen = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        gen = generation_;
    }
    const json content = prompt_content(text, context);
    post([this, gen, content] {
        std::shared_ptr<acp::AcpClient> client;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (gen != generation_ || stop_) return;
            if (session_.empty()) return;   // the session failed to open; its failure is posted
            client = client_;
        }
        if (caliper_builder::Error e = client->prompt_blocks(content)) {
            WidgetAgentEvent ev;
            ev.kind = WidgetAgentEvent::Kind::failed;
            ev.error = e;
            push(std::move(ev));
        }
    });
}

void AcpWidgetAgent::cancel() {
    std::shared_ptr<acp::AcpClient> client;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        client = client_;
    }
    post([client] {
        if (client && client->turn_active()) client->cancel();
    });
}

void AcpWidgetAgent::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++generation_;
        session_.clear();
        events_.clear();
    }
    post([this] {
        std::shared_ptr<acp::AcpClient> client;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            client = client_;
        }
        if (!client || !client->agent_running() || client->session_id().empty()) return;
        if (client->turn_active()) {
            client->cancel();
            // The turn's end arrives as an event; wait for it within the
            // cancel timeout before closing.
            for (int i = 0; i < options_.cancel_timeout_ms / 20 && client->turn_active(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        (void)client->close_session();
    });
}

bool AcpWidgetAgent::poll(WidgetAgentEvent* out) {
    std::shared_ptr<acp::AcpClient> client;
    std::string session;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!events_.empty()) {
            *out = std::move(events_.front());
            events_.pop_front();
            return true;
        }
        client = client_;
        session = session_;
    }
    if (!client) return false;
    acp::AcpEvent ev;
    while (client->poll(&ev)) {
        using K = acp::AcpEventKind;
        if (ev.kind == K::agent_exited) {
            WidgetAgentEvent e;
            e.kind = WidgetAgentEvent::Kind::exited;
            e.error = ev.error ? ev.error : caliper_builder::Error("acp.agent_exited", "The agent stopped.", "");
            *out = std::move(e);
            return true;
        }
        // Events of a session the surface has closed are dropped.
        if (session.empty() || (!ev.session_id.empty() && ev.session_id != session)) continue;
        WidgetAgentEvent e;
        switch (ev.kind) {
            case K::agent_message_chunk:
                if (ev.content_type != "text") continue;
                e.kind = WidgetAgentEvent::Kind::message;
                e.text = ev.text;
                break;
            case K::agent_thought_chunk:
                if (ev.content_type != "text") continue;
                e.kind = WidgetAgentEvent::Kind::thought;
                e.text = ev.text;
                break;
            case K::tool_call:
            case K::tool_call_update:
                if (!ev.tool_call.has_title || ev.tool_call.title.empty()) continue;
                e.kind = WidgetAgentEvent::Kind::progress;
                e.text = ev.tool_call.title;
                break;
            case K::permission_request: {
                // A widget session asks for none, since the preview card is
                // the review; one that comes anyway is declined.
                std::string reject;
                for (const auto& o : ev.options)
                    if (o.kind == "reject_once" || (reject.empty() && o.kind == "reject_always")) reject = o.id;
                if (!reject.empty()) client->respond_permission(ev.permission_id, reject);
                continue;
            }
            case K::turn_ended: {
                if (ev.stop_reason.empty() && ev.error) {
                    e.kind = WidgetAgentEvent::Kind::failed;
                    e.error = ev.error;
                    break;
                }
                e.kind = WidgetAgentEvent::Kind::finished;
                e.stop_reason = ev.stop_reason;
                const json w = ev.raw.is_object() && ev.raw.contains("_meta") && ev.raw["_meta"].is_object() &&
                                       ev.raw["_meta"].contains("caliper") && ev.raw["_meta"]["caliper"].is_object()
                                   ? ev.raw["_meta"]["caliper"].value("widget", json())
                                   : json();
                if (w.is_object()) {
                    if (w.contains("edits") && w["edits"].is_number_integer()) e.edits = w["edits"].get<int>();
                    e.limit = w.value("limit", false) == true;
                    const json h = w.value("handoff", json());
                    if (h.is_object())
                        e.handoff = WidgetHandoff{h.value("kind", std::string("outside_scope")), h.value("reason", std::string())};
                }
                break;
            }
            case K::stalled:
                e.kind = WidgetAgentEvent::Kind::progress;
                e.text = "stalled";
                break;
            default: continue;
        }
        *out = std::move(e);
        return true;
    }
    return false;
}

}  // namespace caliper_host
