// caliper-acp-harness: runs one ACP session headless and prints its transcript
// as JSON Lines on stdout. It starts the agent, sends initialize and
// session/new (with the MCP servers given), sends one prompt, answers every
// permission request with a fixed policy and stops when the turn ends. It
// opens no window and never prints an environment value.
//
// Exit codes: 0 the turn ended with a stop reason; 2 bad usage; 3 the agent
// could not be started, initialised or given a session; 4 the turn failed
// (the agent crashed or answered with an error); 5 the time limit passed;
// 130 interrupted.
#include <caliper_builder/acp/acp_client.h>
#include <caliper_builder/acp/agents.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using caliper_builder::Error;
using namespace caliper_builder::acp;
using nlohmann::json;

namespace {

std::atomic<bool> g_interrupted{false};

extern "C" void on_interrupt(int) { g_interrupted = true; }

void emit(const json& j) {
    std::cout << j.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
    std::cout.flush();
}

json error_json(const Error& e) {
    json j = {{"code", e.code}, {"message", e.message}};
    if (!e.where.empty()) j["where"] = e.where;
    return j;
}

const char* kUsage =
    "Usage: caliper-acp-harness [options] [-- <agent command and arguments>]\n"
    "\n"
    "Runs one ACP session against an agent and prints the transcript as JSON Lines.\n"
    "\n"
    "  --agent <command>          the agent's command and arguments, split into words as a\n"
    "                             shell would split them (no shell runs)\n"
    "  --preset <id>              a known agent: claude, codex or gemini\n"
    "  --cwd <folder>             the session's working folder (default: the current folder)\n"
    "  --mcp <name>=<command>     an MCP server for the agent to start; repeatable\n"
    "  --env NAME=VALUE | NAME    a variable for the agent's environment; NAME alone copies\n"
    "                             this process's value; repeatable; values are never printed\n"
    "  --prompt <text>            the prompt to send (required)\n"
    "  --permission allow|deny|first\n"
    "                             how to answer permission requests (default: deny)\n"
    "  --timeout <seconds>        the time limit for the whole session (default: 300)\n"
    "  --stall <seconds>          report a turn that is silent this long (default: 120)\n"
    "  --help                     this text\n";

int usage_error(const std::string& message) {
    std::cerr << "caliper-acp-harness: " << message << "\n\n" << kUsage;
    return 2;
}

const PermissionOption* choose_option(const std::vector<PermissionOption>& options, const std::string& policy) {
    if (options.empty()) return nullptr;
    if (policy == "first") return &options.front();
    const char* first_kind = policy == "allow" ? "allow_once" : "reject_once";
    const char* second_kind = policy == "allow" ? "allow_always" : "reject_always";
    for (const char* kind : {first_kind, second_kind})
        for (const auto& o : options)
            if (o.kind == kind) return &o;
    return nullptr;
}

void print_stderr_tail(const AcpClient& client) {
    std::string log = client.stderr_log();
    if (log.empty()) return;
    if (log.size() > 4096) log = log.substr(log.size() - 4096);
    std::cerr << "agent stderr (last " << log.size() << " bytes):\n" << log;
    if (log.back() != '\n') std::cerr << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> agent_argv;
    std::string agent_line;
    std::string preset;
    std::string cwd;
    std::string prompt;
    std::string policy = "deny";
    double timeout_s = 300;
    double stall_s = 120;
    std::vector<McpServer> mcp;
    std::vector<EnvVar> env;
    bool have_prompt = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](std::string* out) {
            if (i + 1 >= argc) return false;
            *out = argv[++i];
            return true;
        };
        std::string v;
        if (arg == "--help" || arg == "-h") {
            std::cout << kUsage;
            return 0;
        } else if (arg == "--") {
            for (++i; i < argc; ++i) agent_argv.emplace_back(argv[i]);
        } else if (arg == "--agent") {
            if (!value(&agent_line)) return usage_error("--agent needs a command.");
        } else if (arg == "--preset") {
            if (!value(&preset)) return usage_error("--preset needs an id.");
        } else if (arg == "--cwd") {
            if (!value(&cwd)) return usage_error("--cwd needs a folder.");
        } else if (arg == "--prompt") {
            if (!value(&prompt)) return usage_error("--prompt needs text.");
            have_prompt = true;
        } else if (arg == "--permission") {
            if (!value(&policy) || (policy != "allow" && policy != "deny" && policy != "first"))
                return usage_error("--permission takes allow, deny or first.");
        } else if (arg == "--timeout" || arg == "--stall") {
            if (!value(&v)) return usage_error(arg + " needs a number of seconds.");
            char* end = nullptr;
            const double d = std::strtod(v.c_str(), &end);
            if (end == v.c_str() || *end != '\0' || !(d > 0) || d > 86400)
                return usage_error(arg + " takes a number of seconds between 0 and 86400.");
            (arg == "--timeout" ? timeout_s : stall_s) = d;
        } else if (arg == "--mcp") {
            if (!value(&v)) return usage_error("--mcp needs name=command.");
            const auto eq = v.find('=');
            if (eq == std::string::npos || eq == 0) return usage_error("--mcp takes name=command.");
            std::vector<std::string> words;
            Error e = split_command_line(v.substr(eq + 1), &words);
            if (e) return usage_error("--mcp " + v.substr(0, eq) + ": " + e.message);
            McpServer s;
            s.name = v.substr(0, eq);
            s.command = words[0];
            s.args.assign(words.begin() + 1, words.end());
            mcp.push_back(std::move(s));
        } else if (arg == "--env") {
            if (!value(&v)) return usage_error("--env needs NAME=VALUE or NAME.");
            const auto eq = v.find('=');
            EnvVar ev;
            ev.name = v.substr(0, eq);
            if (ev.name.empty()) return usage_error("--env needs a variable name.");
            if (eq != std::string::npos) {
                ev.value = v.substr(eq + 1);
            } else {
                const char* own = std::getenv(ev.name.c_str());
                if (!own) return usage_error("--env " + ev.name + ": the variable is not set here.");
                ev.value = own;
            }
            env.push_back(std::move(ev));
        } else {
            return usage_error("Unknown option " + arg + ".");
        }
    }

    const int given = (!agent_argv.empty()) + (!agent_line.empty()) + (!preset.empty());
    if (given != 1) return usage_error("Give the agent exactly once: --agent, --preset or -- <command>.");
    if (!agent_line.empty()) {
        Error e = split_command_line(agent_line, &agent_argv);
        if (e) return usage_error("--agent: " + e.message);
    } else if (!preset.empty()) {
        const AgentPreset* p = find_agent_preset(preset);
        if (!p) return usage_error("No known agent is called " + preset + ".");
        agent_argv = p->argv;
    }
    if (!have_prompt) return usage_error("--prompt is required.");

    std::error_code ec;
    std::filesystem::path folder = cwd.empty() ? std::filesystem::current_path(ec) : std::filesystem::path(cwd);
    folder = std::filesystem::weakly_canonical(std::filesystem::absolute(folder, ec), ec);
    if (ec || !std::filesystem::is_directory(folder, ec)) return usage_error("--cwd must name a folder.");

    std::signal(SIGINT, on_interrupt);
    std::signal(SIGTERM, on_interrupt);

    AcpClientOptions options;
    options.agent_argv = agent_argv;
    options.agent_env = env;
    options.agent_cwd = folder.string();
    options.client_name = "caliper-acp-harness";
    options.client_title = "Caliper ACP harness";
    options.stall_timeout_ms = static_cast<int>(stall_s * 1000);
    AcpClient client(options);

    {
        json names = json::array();
        for (const auto& e : env) names.push_back(e.name);
        json servers = json::array();
        for (const auto& s : mcp) servers.push_back({{"name", s.name}, {"command", s.command}, {"args", s.args}});
        emit({{"type", "harness"},
              {"agent", agent_argv},
              {"env_names", names},
              {"cwd", folder.string()},
              {"mcp_servers", servers},
              {"permission", policy}});
    }

    AgentInfo info;
    Error e = client.start(&info);
    if (e) {
        emit({{"type", "error"}, {"stage", "initialize"}, {"error", error_json(e)}});
        print_stderr_tail(client);
        return 3;
    }
    emit({{"type", "initialized"},
          {"protocol_version", info.protocol_version},
          {"agent", {{"name", info.name}, {"title", info.title}, {"version", info.version}}},
          {"capabilities", info.capabilities},
          {"auth_methods", info.auth_methods}});

    SessionInfo session;
    e = client.new_session(folder.string(), mcp, &session);
    if (e) {
        emit({{"type", "error"}, {"stage", "session/new"}, {"error", error_json(e)}});
        print_stderr_tail(client);
        client.shutdown();
        return 3;
    }
    emit({{"type", "session"}, {"session_id", session.id}, {"modes", session.modes}});

    e = client.prompt(prompt);
    if (e) {
        emit({{"type", "error"}, {"stage", "session/prompt"}, {"error", error_json(e)}});
        client.shutdown();
        return 4;
    }
    emit({{"type", "prompt"}, {"text", prompt}});

    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000));
    bool timed_out = false;
    bool interrupted = false;
    bool turn_done = false;
    bool exited = false;
    AcpEvent end;
    clock::time_point hard_stop{};

    while (!turn_done) {
        const auto now = clock::now();
        if ((g_interrupted || now >= deadline) && hard_stop == clock::time_point{}) {
            interrupted = g_interrupted;
            timed_out = !interrupted;
            emit({{"type", interrupted ? "interrupted" : "timeout"}});
            client.cancel();
            hard_stop = now + std::chrono::milliseconds(options.cancel_timeout_ms + 2000);
        }
        if (hard_stop != clock::time_point{} && now >= hard_stop) break;

        AcpEvent ev;
        if (client.wait(&ev, 100)) {
            emit(event_to_json(ev));
            if (ev.kind == AcpEventKind::permission_request) {
                const PermissionOption* o = choose_option(ev.options, policy);
                if (o) {
                    client.respond_permission(ev.permission_id, o->id);
                    emit({{"type", "permission_answer"},
                          {"permission_id", ev.permission_id},
                          {"outcome", "selected"},
                          {"option_id", o->id},
                          {"option_kind", o->kind}});
                } else {
                    emit({{"type", "permission_answer"},
                          {"permission_id", ev.permission_id},
                          {"outcome", "cancelled"}});
                    client.cancel();
                }
            } else if (ev.kind == AcpEventKind::turn_ended) {
                end = ev;
                turn_done = true;
            } else if (ev.kind == AcpEventKind::agent_exited) {
                exited = true;
                end = ev;
                turn_done = true;
            }
        }
    }

    // A crash ends the turn first; wait briefly for the exit status too.
    if (turn_done && !exited && end.error.code == "acp.agent_exited") {
        AcpEvent ev;
        const auto until = clock::now() + std::chrono::milliseconds(1500);
        while (clock::now() < until && client.wait(&ev, 100)) {
            emit(event_to_json(ev));
            if (ev.kind == AcpEventKind::agent_exited) break;
        }
    }
    client.shutdown();

    int code = 0;
    json result = {{"type", "result"}};
    if (interrupted) {
        code = 130;
    } else if (timed_out) {
        code = 5;
    } else if (!turn_done || end.kind != AcpEventKind::turn_ended || end.error) {
        code = 4;
    }
    result["ok"] = code == 0;
    if (turn_done && end.kind == AcpEventKind::turn_ended) result["stop_reason"] = end.stop_reason;
    if (end.error) result["error"] = error_json(end.error);
    result["exit_code"] = code;
    emit(result);
    if (code != 0) print_stderr_tail(client);
    return code;
}
