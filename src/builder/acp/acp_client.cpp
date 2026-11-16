#include <caliper_builder/acp/acp_client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <thread>

namespace caliper_builder::acp {

using nlohmann::json;
using clock_type = std::chrono::steady_clock;

namespace {

constexpr std::size_t kMaxIdBytes = 1024;
constexpr std::size_t kMaxOptions = 64;

const char* const kDeclinedMethods[] = {
    "fs/read_text_file", "fs/write_text_file", "terminal/create",      "terminal/output",
    "terminal/release",  "terminal/kill",      "terminal/wait_for_exit", "elicitation/create",
};

bool get_string(const json& obj, const char* key, std::string* out) {
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_string()) return false;
    *out = it->get<std::string>();
    return true;
}

bool get_bool(const json& obj, const char* key) {
    auto it = obj.find(key);
    return it != obj.end() && it->is_boolean() && it->get<bool>();
}

const json& get_object(const json& obj, const char* key) {
    static const json empty = json::object();
    auto it = obj.find(key);
    return (it != obj.end() && it->is_object()) ? *it : empty;
}

void parse_content_block(const json& block, std::string* type, std::string* text) {
    if (!block.is_object()) return;
    get_string(block, "type", type);
    if (*type == "text") get_string(block, "text", text);
}

// Reads a ToolCall (is_update false) or a ToolCallUpdate. Returns false when a
// required field is missing or has the wrong type.
bool parse_tool_call(const json& obj, bool is_update, ToolCallInfo* out) {
    if (!obj.is_object()) return false;
    if (!get_string(obj, "toolCallId", &out->id) || out->id.empty() || out->id.size() > kMaxIdBytes) return false;
    out->has_title = get_string(obj, "title", &out->title);
    if (!is_update && !out->has_title) return false;
    out->has_kind = get_string(obj, "kind", &out->kind);
    out->has_status = get_string(obj, "status", &out->status);
    if (!is_update) {
        if (!out->has_kind) out->kind = "other";
        if (!out->has_status) out->status = "pending";
    }
    auto content = obj.find("content");
    if (content != obj.end() && content->is_array()) {
        out->has_content = true;
        for (const auto& item : *content) {
            if (!item.is_object()) continue;
            ToolCallContent c;
            get_string(item, "type", &c.type);
            if (c.type == "content") {
                auto block = item.find("content");
                if (block != item.end()) parse_content_block(*block, &c.content_type, &c.text);
            } else if (c.type == "diff") {
                if (!get_string(item, "path", &c.path) || !get_string(item, "newText", &c.new_text)) continue;
                c.has_old_text = get_string(item, "oldText", &c.old_text);
            } else if (c.type == "terminal") {
                get_string(item, "terminalId", &c.terminal_id);
            }
            out->content.push_back(std::move(c));
        }
    }
    auto locations = obj.find("locations");
    if (locations != obj.end() && locations->is_array()) {
        out->has_locations = true;
        for (const auto& item : *locations) {
            if (!item.is_object()) continue;
            ToolLocation l;
            if (!get_string(item, "path", &l.path)) continue;
            auto line = item.find("line");
            if (line != item.end() && line->is_number_integer()) l.line = line->get<int>();
            out->locations.push_back(std::move(l));
        }
    }
    if (obj.contains("rawInput")) out->raw_input = obj["rawInput"];
    if (obj.contains("rawOutput")) out->raw_output = obj["rawOutput"];
    return true;
}

json tool_call_to_json(const ToolCallInfo& t) {
    json j = {{"id", t.id}};
    if (t.has_title) j["title"] = t.title;
    if (t.has_kind || !t.kind.empty()) j["kind"] = t.kind;
    if (t.has_status || !t.status.empty()) j["status"] = t.status;
    if (t.has_content) {
        json items = json::array();
        for (const auto& c : t.content) {
            json i = {{"type", c.type}};
            if (c.type == "content") {
                i["content_type"] = c.content_type;
                if (c.content_type == "text") i["text"] = c.text;
            } else if (c.type == "diff") {
                i["path"] = c.path;
                i["old_text"] = c.has_old_text ? json(c.old_text) : json(nullptr);
                i["new_text"] = c.new_text;
            } else if (c.type == "terminal") {
                i["terminal_id"] = c.terminal_id;
            }
            items.push_back(std::move(i));
        }
        j["content"] = std::move(items);
    }
    if (t.has_locations) {
        json locs = json::array();
        for (const auto& l : t.locations) {
            json o = {{"path", l.path}};
            if (l.line > 0) o["line"] = l.line;
            locs.push_back(std::move(o));
        }
        j["locations"] = std::move(locs);
    }
    if (!t.raw_input.is_null()) j["raw_input"] = t.raw_input;
    if (!t.raw_output.is_null()) j["raw_output"] = t.raw_output;
    return j;
}

json error_to_json(const Error& e) {
    json j = {{"code", e.code}, {"message", e.message}};
    if (!e.where.empty()) j["where"] = e.where;
    return j;
}

}  // namespace

const char* event_kind_name(AcpEventKind kind) {
    switch (kind) {
        case AcpEventKind::user_message_chunk: return "user_message_chunk";
        case AcpEventKind::agent_message_chunk: return "agent_message_chunk";
        case AcpEventKind::agent_thought_chunk: return "agent_thought_chunk";
        case AcpEventKind::tool_call: return "tool_call";
        case AcpEventKind::tool_call_update: return "tool_call_update";
        case AcpEventKind::plan: return "plan";
        case AcpEventKind::available_commands: return "available_commands";
        case AcpEventKind::current_mode: return "current_mode";
        case AcpEventKind::other_update: return "other_update";
        case AcpEventKind::permission_request: return "permission_request";
        case AcpEventKind::permission_withdrawn: return "permission_withdrawn";
        case AcpEventKind::turn_ended: return "turn_ended";
        case AcpEventKind::protocol_fault: return "protocol_fault";
        case AcpEventKind::stalled: return "stalled";
        case AcpEventKind::agent_exited: return "agent_exited";
    }
    return "unknown";
}

json event_to_json(const AcpEvent& ev) {
    json j = {{"type", event_kind_name(ev.kind)}};
    if (!ev.session_id.empty()) j["session_id"] = ev.session_id;
    switch (ev.kind) {
        case AcpEventKind::user_message_chunk:
        case AcpEventKind::agent_message_chunk:
        case AcpEventKind::agent_thought_chunk:
            j["content_type"] = ev.content_type;
            if (ev.content_type == "text") j["text"] = ev.text;
            break;
        case AcpEventKind::tool_call:
        case AcpEventKind::tool_call_update:
            j["tool_call"] = tool_call_to_json(ev.tool_call);
            break;
        case AcpEventKind::plan: {
            json entries = json::array();
            for (const auto& e : ev.plan)
                entries.push_back({{"content", e.content}, {"priority", e.priority}, {"status", e.status}});
            j["entries"] = std::move(entries);
            break;
        }
        case AcpEventKind::available_commands: {
            json cmds = json::array();
            for (const auto& c : ev.commands) {
                json o = {{"name", c.name}, {"description", c.description}};
                if (!c.hint.empty()) o["hint"] = c.hint;
                cmds.push_back(std::move(o));
            }
            j["commands"] = std::move(cmds);
            break;
        }
        case AcpEventKind::current_mode:
            j["mode_id"] = ev.mode_id;
            break;
        case AcpEventKind::other_update:
            j["update_kind"] = ev.update_kind;
            j["update"] = ev.raw;
            break;
        case AcpEventKind::permission_request: {
            j["permission_id"] = ev.permission_id;
            j["tool_call"] = tool_call_to_json(ev.tool_call);
            json opts = json::array();
            for (const auto& o : ev.options) opts.push_back({{"id", o.id}, {"name", o.name}, {"kind", o.kind}});
            j["options"] = std::move(opts);
            break;
        }
        case AcpEventKind::permission_withdrawn:
            j["permission_id"] = ev.permission_id;
            break;
        case AcpEventKind::turn_ended:
            j["stop_reason"] = ev.stop_reason;
            break;
        case AcpEventKind::protocol_fault:
        case AcpEventKind::stalled:
            break;
        case AcpEventKind::agent_exited:
            if (ev.has_exit_status) {
                j["exited"] = ev.exit_status.exited;
                if (ev.exit_status.exited) j["exit_code"] = ev.exit_status.code;
                else j["signal"] = ev.exit_status.signal;
            }
            break;
    }
    if (ev.error) j["error"] = error_to_json(ev.error);
    return j;
}

// ---- state ------------------------------------------------------------------

struct AcpClient::State {
    AcpClientOptions options;
    std::unique_ptr<Process> process;
    std::unique_ptr<JsonRpcPeer> peer;
    std::thread stderr_thread;
    std::atomic<bool> shutting_down{false};

    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<AcpEvent> events;
    std::size_t dropped = 0;
    AgentInfo agent;
    bool started = false;
    bool agent_gone = false;
    std::string session_id;
    bool turn_active = false;
    std::uint64_t turn_seq = 0;
    std::int64_t prompt_call_id = -1;
    bool cancel_requested = false;
    bool stall_reported = false;
    clock_type::time_point last_action;  // the client's last prompt or answer
    struct OpenPermission {
        json rpc_id;
        std::vector<std::string> option_ids;
    };
    std::map<std::uint64_t, OpenPermission> permissions;
    std::uint64_t next_permission_id = 1;

    mutable std::mutex log_mutex;
    std::string stderr_log;

    // Callers hold `mutex`.
    void push_locked(AcpEvent ev) {
        if (events.size() >= options.max_events && !events.empty()) {
            // Drop the oldest chunk, so tool calls, permission requests and the
            // end of a turn are kept; fall back to the oldest event.
            auto it = std::find_if(events.begin(), events.end(), [](const AcpEvent& e) {
                return e.kind == AcpEventKind::agent_message_chunk || e.kind == AcpEventKind::agent_thought_chunk ||
                       e.kind == AcpEventKind::user_message_chunk || e.kind == AcpEventKind::other_update;
            });
            events.erase(it != events.end() ? it : events.begin());
            ++dropped;
        }
        events.push_back(std::move(ev));
        cv.notify_all();
    }

    void push(AcpEvent ev) {
        std::lock_guard<std::mutex> lock(mutex);
        push_locked(std::move(ev));
    }

    void fault(const std::string& code, const std::string& message, const std::string& where = {}) {
        AcpEvent ev;
        ev.kind = AcpEventKind::protocol_fault;
        ev.error = {code, message, where};
        push(std::move(ev));
    }

    void stderr_loop() {
        std::vector<char> buf(4096);
        while (process) {
            std::size_t got = 0;
            const IoStatus st = process->read_stderr(buf.data(), buf.size(), &got, 50);
            if (st == IoStatus::ok) {
                std::lock_guard<std::mutex> lock(log_mutex);
                stderr_log.append(buf.data(), got);
                const std::size_t cap = options.stderr_log_bytes;
                if (stderr_log.size() > cap) stderr_log.erase(0, stderr_log.size() - cap);
            } else if (st == IoStatus::eof || st == IoStatus::error) {
                break;
            } else if (shutting_down && !process_running()) {
                break;
            }
        }
    }

    bool process_running() {
        ExitStatus s;
        return process && !process->try_wait(&s);
    }

    void on_update(const json& params);
    void on_permission(const json& id, const json& params);
    void on_cancel_request(const json& params);
    void on_prompt_done(std::uint64_t seq, RpcResult r);
    void on_closed(const Error& reason);
    void check_watchdogs();
    void stop_agent();
};

void AcpClient::State::on_update(const json& params) {
    if (!params.is_object()) {
        fault("acp.bad_update", "A session/update without params was dropped.");
        return;
    }
    std::string sid;
    if (!get_string(params, "sessionId", &sid) || sid.size() > kMaxIdBytes) {
        fault("acp.bad_update", "A session/update without a session id was dropped.");
        return;
    }
    auto upd = params.find("update");
    std::string kind;
    if (upd == params.end() || !upd->is_object() || !get_string(*upd, "sessionUpdate", &kind)) {
        fault("acp.bad_update", "A session/update without an update kind was dropped.");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!session_id.empty() && sid != session_id) {
            // Fall through to the fault below, outside the lock.
            sid.clear();
        }
    }
    if (sid.empty()) {
        fault("acp.bad_update", "A session/update for another session was dropped.", kind);
        return;
    }

    const json& u = *upd;
    AcpEvent ev;
    ev.session_id = sid;
    ev.update_kind = kind;
    if (kind == "user_message_chunk" || kind == "agent_message_chunk" || kind == "agent_thought_chunk") {
        ev.kind = kind == "user_message_chunk"    ? AcpEventKind::user_message_chunk
                  : kind == "agent_message_chunk" ? AcpEventKind::agent_message_chunk
                                                  : AcpEventKind::agent_thought_chunk;
        auto content = u.find("content");
        if (content == u.end() || !content->is_object()) {
            fault("acp.bad_update", "A message chunk without content was dropped.", kind);
            return;
        }
        parse_content_block(*content, &ev.content_type, &ev.text);
        if (ev.content_type != "text") ev.raw = u;
    } else if (kind == "tool_call" || kind == "tool_call_update") {
        ev.kind = kind == "tool_call" ? AcpEventKind::tool_call : AcpEventKind::tool_call_update;
        if (!parse_tool_call(u, kind == "tool_call_update", &ev.tool_call)) {
            fault("acp.bad_update", "A tool call without an id or a title was dropped.", kind);
            return;
        }
    } else if (kind == "plan") {
        ev.kind = AcpEventKind::plan;
        auto entries = u.find("entries");
        if (entries == u.end() || !entries->is_array()) {
            fault("acp.bad_update", "A plan without entries was dropped.", kind);
            return;
        }
        for (const auto& e : *entries) {
            if (!e.is_object()) continue;
            PlanEntry p;
            get_string(e, "content", &p.content);
            get_string(e, "priority", &p.priority);
            get_string(e, "status", &p.status);
            ev.plan.push_back(std::move(p));
        }
    } else if (kind == "available_commands_update") {
        ev.kind = AcpEventKind::available_commands;
        auto cmds = u.find("availableCommands");
        if (cmds == u.end() || !cmds->is_array()) {
            fault("acp.bad_update", "A command list without commands was dropped.", kind);
            return;
        }
        for (const auto& c : *cmds) {
            if (!c.is_object()) continue;
            CommandInfo info;
            if (!get_string(c, "name", &info.name)) continue;
            get_string(c, "description", &info.description);
            const json& input = get_object(c, "input");
            get_string(input, "hint", &info.hint);
            ev.commands.push_back(std::move(info));
        }
    } else if (kind == "current_mode_update") {
        ev.kind = AcpEventKind::current_mode;
        if (!get_string(u, "currentModeId", &ev.mode_id)) {
            fault("acp.bad_update", "A mode change without a mode id was dropped.", kind);
            return;
        }
    } else {
        ev.kind = AcpEventKind::other_update;
        ev.raw = u;
    }
    push(std::move(ev));
}

void AcpClient::State::on_permission(const json& id, const json& params) {
    auto refuse = [&](const std::string& message) {
        peer->respond_error(id, rpc_code::invalid_params, message);
        fault("acp.bad_request", message, "session/request_permission");
    };
    if (!params.is_object()) return refuse("The permission request has no params.");
    std::string sid;
    if (!get_string(params, "sessionId", &sid)) return refuse("The permission request has no session id.");
    AcpEvent ev;
    ev.kind = AcpEventKind::permission_request;
    ev.session_id = sid;
    auto tc = params.find("toolCall");
    if (tc == params.end() || !parse_tool_call(*tc, true, &ev.tool_call))
        return refuse("The permission request has no tool call.");
    auto opts = params.find("options");
    if (opts == params.end() || !opts->is_array() || opts->empty() || opts->size() > kMaxOptions)
        return refuse("The permission request has no options, or too many.");
    std::vector<std::string> ids;
    std::set<std::string> seen;
    for (const auto& o : *opts) {
        PermissionOption p;
        if (!o.is_object() || !get_string(o, "optionId", &p.id) || !get_string(o, "name", &p.name) ||
            !get_string(o, "kind", &p.kind) || p.id.empty() || p.id.size() > kMaxIdBytes || !seen.insert(p.id).second)
            return refuse("A permission option lacks an id, a name or a kind, or repeats an id.");
        ids.push_back(p.id);
        ev.options.push_back(std::move(p));
    }
    ev.raw = params;

    bool answer_cancelled = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!session_id.empty() && sid != session_id) {
            answer_cancelled = false;
        } else if (turn_active && cancel_requested) {
            answer_cancelled = true;
        } else {
            ev.permission_id = next_permission_id++;
            permissions[ev.permission_id] = OpenPermission{id, ids};
            push_locked(std::move(ev));
            return;
        }
    }
    if (answer_cancelled) {
        peer->respond(id, json{{"outcome", {{"outcome", "cancelled"}}}});
        return;
    }
    peer->respond_error(id, rpc_code::resource_not_found, "Unknown session.");
    fault("acp.bad_request", "A permission request for another session was refused.", "session/request_permission");
}

void AcpClient::State::on_cancel_request(const json& params) {
    if (!params.is_object() || !params.contains("requestId")) return;
    const json& rid = params["requestId"];
    std::uint64_t found = 0;
    json rpc_id;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = permissions.begin(); it != permissions.end(); ++it) {
            if (it->second.rpc_id == rid) {
                found = it->first;
                rpc_id = it->second.rpc_id;
                permissions.erase(it);
                break;
            }
        }
        if (!found) return;
        AcpEvent ev;
        ev.kind = AcpEventKind::permission_withdrawn;
        ev.session_id = session_id;
        ev.permission_id = found;
        push_locked(std::move(ev));
    }
    peer->respond(rpc_id, json{{"outcome", {{"outcome", "cancelled"}}}});
}

void AcpClient::State::on_prompt_done(std::uint64_t seq, RpcResult r) {
    std::vector<std::pair<std::uint64_t, json>> stale;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (seq != turn_seq || !turn_active) return;
        turn_active = false;
        prompt_call_id = -1;
        const bool cancelled = cancel_requested;
        cancel_requested = false;
        // A permission request still open when its turn ends will never be used.
        for (auto& [pid, p] : permissions) stale.emplace_back(pid, p.rpc_id);
        permissions.clear();
        for (auto& [pid, rid] : stale) {
            (void)rid;
            AcpEvent w;
            w.kind = AcpEventKind::permission_withdrawn;
            w.session_id = session_id;
            w.permission_id = pid;
            push_locked(std::move(w));
        }

        AcpEvent ev;
        ev.kind = AcpEventKind::turn_ended;
        ev.session_id = session_id;
        if (r.ok()) {
            if (!r.result.is_object() || !get_string(r.result, "stopReason", &ev.stop_reason) ||
                ev.stop_reason.empty() || ev.stop_reason.size() > 64) {
                ev.stop_reason.clear();
                ev.error = {"acp.bad_response", "The agent ended the turn without a stop reason.", "session/prompt"};
            }
            ev.raw = r.result;
        } else if (cancelled && r.error.code == "rpc.timeout") {
            ev.stop_reason = "cancelled";
            ev.error = {"acp.cancel_unanswered",
                        "The agent did not end the turn after it was cancelled; stop the agent to end it.",
                        "session/prompt"};
        } else if (r.error.code == "rpc.closed") {
            ev.error = {"acp.agent_exited", "The agent stopped during the turn.", "session/prompt"};
        } else if (cancelled && r.error.code == "rpc.remote_error") {
            // ACP asks agents to answer a cancelled turn with the cancelled stop
            // reason, not an error; one that errs is treated as cancelled.
            ev.stop_reason = "cancelled";
            ev.raw = {{"error", r.error.message}};
        } else {
            ev.error = {"acp.prompt_failed", "The agent could not run the turn: " + r.error.message, "session/prompt"};
        }
        push_locked(std::move(ev));
    }
    for (auto& [pid, rid] : stale) {
        (void)pid;
        peer->respond(rid, json{{"outcome", {{"outcome", "cancelled"}}}});
    }
}

void AcpClient::State::on_closed(const Error& reason) {
    (void)reason;
    if (shutting_down) {
        std::lock_guard<std::mutex> lock(mutex);
        agent_gone = true;
        permissions.clear();
        return;
    }
    AcpEvent ev;
    ev.kind = AcpEventKind::agent_exited;
    ExitStatus st;
    if (process && process->wait_for(1000, &st)) {
        ev.has_exit_status = true;
        ev.exit_status = st;
        ev.error = {"acp.agent_exited", "The agent exited with " + st.describe() + "."};
    } else {
        ev.error = {"acp.agent_exited", "The agent closed its output but is still running."};
    }
    std::lock_guard<std::mutex> lock(mutex);
    agent_gone = true;
    permissions.clear();
    ev.session_id = session_id;
    push_locked(std::move(ev));
}

void AcpClient::State::check_watchdogs() {
    if (!peer || options.stall_timeout_ms <= 0) return;
    auto last = peer->last_receive();
    std::lock_guard<std::mutex> lock(mutex);
    if (!turn_active || stall_reported || !permissions.empty() || agent_gone) return;
    if (last_action > last) last = last_action;
    if (clock_type::now() - last < std::chrono::milliseconds(options.stall_timeout_ms)) return;
    stall_reported = true;
    AcpEvent ev;
    ev.kind = AcpEventKind::stalled;
    ev.session_id = session_id;
    ev.error = {"acp.stalled", "The agent has sent nothing for " + std::to_string(options.stall_timeout_ms / 1000) +
                                   " seconds in this turn."};
    push_locked(std::move(ev));
}

void AcpClient::State::stop_agent() {
    if (!process) return;
    shutting_down = true;
    if (peer) {
        peer->flush(200);
        peer->close();
    }
    ExitStatus st;
    if (!process->wait_for(std::min(options.kill_grace_ms, 1000), &st)) process->terminate(options.kill_grace_ms);
    if (stderr_thread.joinable()) stderr_thread.join();
    std::lock_guard<std::mutex> lock(mutex);
    agent_gone = true;
    turn_active = false;
}

// ---- the client -------------------------------------------------------------

AcpClient::AcpClient(AcpClientOptions options) : state_(std::make_shared<State>()) {
    state_->options = std::move(options);
    if (state_->options.max_events == 0) state_->options.max_events = 1;
}

AcpClient::~AcpClient() { shutdown(); }

void AcpClient::shutdown() { state_->stop_agent(); }

Error AcpClient::start(AgentInfo* info) {
    State& s = *state_;
    if (s.process) return {"acp.already_started", "The agent has already been started."};
    if (s.options.agent_argv.empty() || s.options.agent_argv[0].empty())
        return {"acp.bad_argument", "No agent command was given."};

    SpawnOptions so;
    so.argv = s.options.agent_argv;
    so.env = s.options.agent_env;
    so.env_remove = s.options.agent_env_remove;
    so.cwd = s.options.agent_cwd;
    so.inherit_environment = true;
    so.new_process_group = true;
    Error e;
    s.process = Process::spawn(so, &e);
    if (!s.process) return e;

    JsonRpcOptions ro;
    ro.max_message_bytes = s.options.max_message_bytes;
    s.peer = std::make_unique<JsonRpcPeer>(make_process_channel(*s.process), ro);
    State* sp = &s;
    s.peer->on_notification("session/update", [sp](const json& p) { sp->on_update(p); });
    s.peer->on_request("session/request_permission", [sp](const json& id, const json& p) { sp->on_permission(id, p); });
    s.peer->on_notification("$/cancel_request", [sp](const json& p) { sp->on_cancel_request(p); });
    for (const char* m : kDeclinedMethods) {
        const std::string method = m;
        s.peer->on_request(method, [sp, method](const json& id, const json&) {
            sp->peer->respond_error(id, rpc_code::method_not_found,
                                    "Method not found: " + method + " (this client declined the capability)");
            sp->fault("acp.declined_method", "The agent called " + method + ", which this client declined.", method);
        });
    }
    s.peer->on_fault([sp](const Error& f) { sp->fault(f.code, f.message, f.where); });
    s.peer->on_close([sp](const Error& reason) { sp->on_closed(reason); });
    s.peer->start();
    s.stderr_thread = std::thread([sp] { sp->stderr_loop(); });

    const json params = {
        {"protocolVersion", kAcpProtocolVersion},
        {"clientCapabilities", {{"fs", {{"readTextFile", false}, {"writeTextFile", false}}}, {"terminal", false}}},
        {"clientInfo",
         {{"name", s.options.client_name}, {"title", s.options.client_title}, {"version", s.options.client_version}}},
    };
    RpcResult r = s.peer->call_sync("initialize", params, s.options.request_timeout_ms);
    auto fail = [&](Error err) {
        s.stop_agent();
        return err;
    };
    if (!r.ok()) {
        if (r.error.code == "rpc.closed")
            return fail({"acp.agent_exited", "The agent exited before it answered initialize.", "initialize"});
        if (r.error.code == "rpc.timeout")
            return fail({"rpc.timeout", "The agent did not answer initialize in time.", "initialize"});
        return fail({"acp.bad_response", "The agent refused initialize: " + r.error.message, "initialize"});
    }
    if (!r.result.is_object()) return fail({"acp.bad_response", "The agent's initialize answer is not an object.", "initialize"});
    auto pv = r.result.find("protocolVersion");
    if (pv == r.result.end() || !pv->is_number_integer())
        return fail({"acp.bad_response", "The agent's initialize answer has no protocol version.", "initialize"});
    const auto version = pv->get<std::int64_t>();
    if (version != kAcpProtocolVersion)
        return fail({"acp.version_unsupported",
                     "The agent speaks ACP version " + std::to_string(version) + "; Caliper speaks version " +
                         std::to_string(kAcpProtocolVersion) + ".",
                     "initialize"});

    AgentInfo a;
    a.protocol_version = static_cast<int>(version);
    const json& caps = get_object(r.result, "agentCapabilities");
    a.capabilities = caps;
    a.load_session = get_bool(caps, "loadSession");
    const json& mcp = get_object(caps, "mcpCapabilities");
    a.mcp_http = get_bool(mcp, "http");
    a.mcp_sse = get_bool(mcp, "sse");
    const json& pc = get_object(caps, "promptCapabilities");
    a.prompt_image = get_bool(pc, "image");
    a.prompt_audio = get_bool(pc, "audio");
    a.prompt_embedded_context = get_bool(pc, "embeddedContext");
    auto am = r.result.find("authMethods");
    a.auth_methods = (am != r.result.end() && am->is_array()) ? *am : json::array();
    const json& ai = get_object(r.result, "agentInfo");
    get_string(ai, "name", &a.name);
    get_string(ai, "title", &a.title);
    get_string(ai, "version", &a.version);
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.agent = a;
        s.started = true;
    }
    if (info) *info = a;
    return {};
}

Error AcpClient::authenticate(const std::string& method_id, int timeout_ms) {
    State& s = *state_;
    if (!s.peer || !s.started) return {"acp.not_started", "The agent has not been started."};
    bool advertised = false;
    for (const auto& m : s.agent.auth_methods) {
        std::string id;
        std::string type = "agent";
        if (m.is_object() && get_string(m, "id", &id) && id == method_id) {
            get_string(m, "type", &type);
            advertised = type == "agent";
        }
    }
    if (!advertised)
        return {"acp.bad_argument", "The agent advertises no login method of type agent with that id.", method_id};
    RpcResult r = s.peer->call_sync("authenticate", json{{"methodId", method_id}}, timeout_ms);
    if (!r.ok()) return {"acp.auth_required", "The agent's login did not complete: " + r.error.message, method_id};
    return {};
}

Error AcpClient::new_session(const std::string& cwd, const std::vector<McpServer>& mcp_servers, SessionInfo* session) {
    State& s = *state_;
    if (!s.peer || !s.started) return {"acp.not_started", "The agent has not been started."};
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.agent_gone) return {"acp.agent_exited", "The agent is no longer running."};
    }
    std::error_code ec;
    if (cwd.empty() || !std::filesystem::path(cwd).is_absolute() || !std::filesystem::is_directory(cwd, ec))
        return {"acp.bad_argument", "The session folder must be an absolute path to a folder.", cwd};

    std::string path_list;
    for (const auto& v : s.options.agent_env)
        if (v.name == "PATH") path_list = v.value;
    if (path_list.empty()) {
        const char* p = std::getenv("PATH");
        path_list = p ? p : "/usr/bin:/bin";
    }
    json servers = json::array();
    std::set<std::string> names;
    for (const auto& m : mcp_servers) {
        if (m.name.empty() || m.name.size() > 256 || !names.insert(m.name).second)
            return {"acp.bad_argument", "Each MCP server needs a name of its own.", m.name};
        if (m.command.empty()) return {"acp.bad_argument", "An MCP server has no command.", m.name};
        std::string command = m.command;
        if (command.find('/') == std::string::npos) {
            command = find_executable(command, path_list);
            if (command.empty())
                return {"acp.bad_argument", "The MCP server command " + m.command + " was not found on PATH.", m.name};
        } else if (!std::filesystem::path(command).is_absolute()) {
            command = std::filesystem::absolute(command, ec).lexically_normal().string();
        }
        json env = json::array();
        for (const auto& v : m.env) env.push_back({{"name", v.name}, {"value", v.value}});
        servers.push_back({{"name", m.name}, {"command", command}, {"args", m.args}, {"env", env}});
    }

    RpcResult r = s.peer->call_sync("session/new", json{{"cwd", cwd}, {"mcpServers", servers}},
                                    s.options.request_timeout_ms);
    if (!r.ok()) {
        if (r.error.code == "rpc.remote_error" && r.rpc_code == rpc_code::auth_required) {
            std::string methods;
            for (const auto& m : s.agent.auth_methods) {
                std::string name;
                if (m.is_object() && get_string(m, "name", &name)) methods += (methods.empty() ? "" : ", ") + name;
            }
            return {"acp.auth_required",
                    "The agent asks for a login. Log in with the agent's own command, then start again" +
                        (methods.empty() ? std::string(".") : " (the agent offers: " + methods + ")."),
                    "session/new"};
        }
        if (r.error.code == "rpc.closed")
            return {"acp.agent_exited", "The agent exited before it answered session/new.", "session/new"};
        if (r.error.code == "rpc.timeout") return r.error;
        return {"acp.bad_response", "The agent refused session/new: " + r.error.message, "session/new"};
    }
    std::string sid;
    if (!r.result.is_object() || !get_string(r.result, "sessionId", &sid) || sid.empty() || sid.size() > kMaxIdBytes)
        return {"acp.bad_response", "The agent's session/new answer has no session id.", "session/new"};
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.session_id = sid;
    }
    if (session) {
        session->id = sid;
        auto modes = r.result.find("modes");
        session->modes = (modes != r.result.end() && modes->is_object()) ? *modes : json();
        auto co = r.result.find("configOptions");
        session->config_options = (co != r.result.end() && co->is_array()) ? *co : json();
    }
    return {};
}

Error AcpClient::prompt(const std::string& text) {
    return prompt_blocks(json::array({{{"type", "text"}, {"text", text}}}));
}

Error AcpClient::prompt_blocks(const json& content_blocks) {
    State& s = *state_;
    if (!content_blocks.is_array() || content_blocks.empty())
        return {"acp.bad_argument", "A prompt needs at least one content block."};
    std::uint64_t seq = 0;
    std::string sid;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.started || !s.peer) return {"acp.not_started", "The agent has not been started."};
        if (s.agent_gone) return {"acp.agent_exited", "The agent is no longer running."};
        if (s.session_id.empty()) return {"acp.no_session", "No session has been created."};
        if (s.turn_active) return {"acp.turn_active", "A turn is already running; cancel it or wait for its end."};
        s.turn_active = true;
        s.cancel_requested = false;
        s.stall_reported = false;
        s.prompt_call_id = -1;
        s.last_action = clock_type::now();
        seq = ++s.turn_seq;
        sid = s.session_id;
    }
    State* sp = &s;
    const std::int64_t id = s.peer->call_async("session/prompt", json{{"sessionId", sid}, {"prompt", content_blocks}},
                                               0, [sp, seq](RpcResult r) { sp->on_prompt_done(seq, std::move(r)); });
    bool cancel_now = false;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.turn_seq == seq && s.turn_active) {
            s.prompt_call_id = id;
            cancel_now = s.cancel_requested;
        }
    }
    if (cancel_now && id >= 0) s.peer->set_call_timeout(id, s.options.cancel_timeout_ms);
    return {};
}

Error AcpClient::cancel() {
    State& s = *state_;
    std::vector<std::pair<std::uint64_t, json>> open;
    std::int64_t call_id = -1;
    std::string sid;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.turn_active) return {"acp.no_turn", "No turn is running."};
        if (s.cancel_requested) return {};
        s.cancel_requested = true;
        call_id = s.prompt_call_id;
        sid = s.session_id;
        for (auto& [pid, p] : s.permissions) open.emplace_back(pid, p.rpc_id);
        s.permissions.clear();
        for (auto& [pid, rid] : open) {
            (void)rid;
            AcpEvent w;
            w.kind = AcpEventKind::permission_withdrawn;
            w.session_id = sid;
            w.permission_id = pid;
            s.push_locked(std::move(w));
        }
    }
    Error e = s.peer->notify("session/cancel", json{{"sessionId", sid}});
    for (auto& [pid, rid] : open) {
        (void)pid;
        s.peer->respond(rid, json{{"outcome", {{"outcome", "cancelled"}}}});
    }
    if (call_id >= 0) s.peer->set_call_timeout(call_id, s.options.cancel_timeout_ms);
    return e;
}

Error AcpClient::respond_permission(std::uint64_t permission_id, const std::string& option_id) {
    State& s = *state_;
    json rpc_id;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.permissions.find(permission_id);
        if (it == s.permissions.end())
            return {"acp.unknown_permission", "No open permission request has that id.",
                    std::to_string(permission_id)};
        const auto& ids = it->second.option_ids;
        if (std::find(ids.begin(), ids.end(), option_id) == ids.end())
            return {"acp.unknown_option", "The permission request offers no option with that id.", option_id};
        rpc_id = it->second.rpc_id;
        s.permissions.erase(it);
        s.last_action = clock_type::now();
    }
    return s.peer->respond(rpc_id, json{{"outcome", {{"outcome", "selected"}, {"optionId", option_id}}}});
}

bool AcpClient::poll(AcpEvent* event) {
    State& s = *state_;
    s.check_watchdogs();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.dropped > 0) {
        event->kind = AcpEventKind::protocol_fault;
        event->session_id = s.session_id;
        event->error = {"acp.events_dropped",
                        std::to_string(s.dropped) + " events were dropped because nobody read the queue."};
        s.dropped = 0;
        return true;
    }
    if (s.events.empty()) return false;
    *event = std::move(s.events.front());
    s.events.pop_front();
    return true;
}

bool AcpClient::wait(AcpEvent* event, int timeout_ms) {
    State& s = *state_;
    const auto deadline = clock_type::now() + std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
    for (;;) {
        if (poll(event)) return true;
        const auto now = clock_type::now();
        if (timeout_ms >= 0 && now >= deadline) return false;
        auto slice = std::chrono::milliseconds(100);
        if (timeout_ms >= 0) slice = std::min(slice, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now) +
                                                         std::chrono::milliseconds(1));
        std::unique_lock<std::mutex> lock(s.mutex);
        s.cv.wait_for(lock, slice, [&] { return !s.events.empty() || s.dropped > 0; });
    }
}

bool AcpClient::turn_active() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->turn_active;
}

bool AcpClient::agent_running() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->process && !state_->agent_gone;
}

const AgentInfo& AcpClient::agent() const { return state_->agent; }

std::string AcpClient::session_id() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->session_id;
}

std::string AcpClient::stderr_log() const {
    std::lock_guard<std::mutex> lock(state_->log_mutex);
    return state_->stderr_log;
}

std::string AcpClient::describe_command() const {
    SpawnOptions so;
    so.argv = state_->options.agent_argv;
    so.env = state_->options.agent_env;
    so.env_remove = state_->options.agent_env_remove;
    so.cwd = state_->options.agent_cwd;
    return describe_spawn(so);
}

}  // namespace caliper_builder::acp
