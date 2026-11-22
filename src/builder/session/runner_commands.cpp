// The model runner's commands (runner_commands.h, design/hosted-runs.md).
//
// The handlers check what needs no child process: the script is a file in the
// workspace, the id is valid, the parameters are few and named, and a start
// or a fork was approved on its card. The runner then starts the process, or
// sends the checkpoint or the stop action, and answers at once.
#include <caliper_builder/session/runner_commands.h>

#include <caliper_builder/loaders/paths.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <system_error>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;
namespace fs = std::filesystem;

Error fail(std::string code, std::string message, std::string where = {}) {
    return Error(std::move(code), std::move(message), std::move(where));
}

std::string str_arg(const json& args, const char* key) {
    const auto it = args.find(key);
    return it != args.end() && it->is_string() ? it->get<std::string>() : std::string();
}

// A text cut to `limit` bytes at a character's start, with "..." when cut.
std::string cut_text(const std::string& s, std::size_t limit) {
    if (s.size() <= limit) return s;
    std::size_t end = limit;
    while (end > 0 && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) --end;
    return s.substr(0, end) + "...";
}

// Whether a bare word reads as a number, true or false in the Parameters
// field of the runs panel, so that a string of that form needs quotes.
bool reads_as_scalar(const std::string& v) {
    if (v.empty()) return false;
    if (v == "true" || v == "false") return true;
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    return end && *end == '\0' && std::isfinite(d);
}

std::string value_text(const json& v) {
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        const bool plain = !s.empty() && s.find_first_of(" \t,\"") == std::string::npos && !reads_as_scalar(s);
        return plain ? s : "\"" + s + "\"";
    }
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    return v.dump();
}

std::string arg_text(const std::string& a) {
    return a.empty() || a.find_first_of(" \t") != std::string::npos ? "\"" + a + "\"" : a;
}

constexpr std::size_t kCardTextLimit = 300;   // bytes of the arguments, and of the parameters, on a card

// A parameter name as caliper.param takes it: 1 to 63 bytes, no control
// characters.
bool valid_param_name(const std::string& name) {
    if (name.empty() || name.size() > 63) return false;
    for (const unsigned char ch : name)
        if (ch < 0x20 || ch == 0x7f) return false;
    return true;
}

// The `params` argument, a list of {name, value}, as an object of values by
// name, or an Error. The schema checked the count, the names and the types.
Error params_arg(const json& args, json* out) {
    *out = json::object();
    const auto it = args.find("params");
    if (it == args.end()) return {};
    if (!it->is_array() || it->size() > kRunMaxParams)
        return fail("command.invalid_args", "params is a list of at most 64 {name, value} pairs.", "args.params");
    for (const auto& p : *it) {
        const std::string name = p.is_object() ? str_arg(p, "name") : std::string();
        if (!valid_param_name(name))
            return fail("command.invalid_args",
                        "A parameter name has 1 to 63 bytes and no control characters, as caliper.param takes it.",
                        "args.params");
        const json v = p.value("value", json());
        if (!(v.is_number() || v.is_boolean() || v.is_string()))
            return fail("command.invalid_args", "The parameter " + name + " takes a number, true or false, or a string.",
                        "args.params");
        if (out->contains(name))
            return fail("command.invalid_args", "The parameter " + name + " is named twice.", "args.params");
        (*out)[name] = v;
    }
    return {};
}

// The `id` argument: empty when absent, else a valid id or an Error.
Error id_arg(const json& args, std::string* out) {
    *out = str_arg(args, "id");
    if (args.contains("id") && !valid_run_id(*out))
        return fail(kRunBadId, "A run id has 1 to 40 characters, each a-z, 0-9 or -.", "args.id");
    return {};
}

Error not_approved(const std::string& name) {
    return fail("command.not_allowed",
                name + " runs a Python script, so it runs only after the person approves it on its card.", name);
}

// The script argument as a canonical file in the workspace.
Error script_arg(const json& args, const std::string& workspace, RunStartRequest* out) {
    const std::string script = str_arg(args, "script");
    const std::string refused = "The script " + script + " is not a file in the workspace " + workspace +
                                "; a managed run starts only a script of the workspace.";
    std::error_code ec;
    fs::path p = fs::u8path(script);
    if (p.is_relative()) p = fs::u8path(workspace) / p;
    const fs::path canon = fs::canonical(p, ec);
    if (ec || script.empty()) return fail(kRunScriptRefused, refused, "args.script");
    const std::string canon_text = canon.u8string();
    if (!path_within(canon_text, workspace) || !fs::is_regular_file(canon, ec) || ec)
        return fail(kRunScriptRefused, refused, "args.script");
    out->script_path = canon_text;
    out->script = canon.lexically_relative(fs::u8path(workspace)).generic_u8string();
    return {};
}

CommandOutput start(const json& args, ModelRunner& runner, const std::string& workspace, bool approved) {
    RunStartRequest req;
    if (Error e = script_arg(args, workspace, &req)) return CommandOutput::fail(std::move(e));
    if (const auto it = args.find("args"); it != args.end() && it->is_array()) {
        if (it->size() > kRunMaxArgs)
            return CommandOutput::fail(fail("command.invalid_args", "A run takes at most 64 arguments.", "args.args"));
        for (const auto& a : *it) {
            if (!a.is_string())
                return CommandOutput::fail(fail("command.invalid_args", "Each argument is a string.", "args.args"));
            req.args.push_back(a.get<std::string>());
        }
    }
    if (Error e = params_arg(args, &req.params)) return CommandOutput::fail(std::move(e));
    if (Error e = id_arg(args, &req.id)) return CommandOutput::fail(std::move(e));
    if (!approved) return CommandOutput::fail(not_approved("run.start"));
    RunEntry entry;
    if (Error e = runner.start(req, &entry)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"run", run_entry_json(entry)}});
}

CommandOutput fork(const json& args, ModelRunner& runner, bool approved) {
    RunForkRequest req;
    req.parent = str_arg(args, "run");
    if (req.parent == kUserRunId)
        return CommandOutput::fail(fail(kRunNotManaged,
                                        "The run user is the person's own script, and only a managed run can fork; "
                                        "start the script with run.start first.",
                                        "args.run"));
    if (Error e = params_arg(args, &req.params)) return CommandOutput::fail(std::move(e));
    if (Error e = id_arg(args, &req.id)) return CommandOutput::fail(std::move(e));
    if (!approved) return CommandOutput::fail(not_approved("run.fork"));
    RunEntry entry;
    if (Error e = runner.fork(req, &entry)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"run", run_entry_json(entry)}});
}

CommandOutput stop(const json& args, ModelRunner& runner) {
    const std::string run = str_arg(args, "run");
    if (run == kUserRunId)
        return CommandOutput::fail(fail(kRunNotManaged,
                                        "The run user is the person's own script, which the person stops; run.stop "
                                        "stops a managed run.",
                                        "args.run"));
    double after = 10.0;
    if (const auto it = args.find("kill_after_s"); it != args.end() && it->is_number()) after = it->get<double>();
    if (!(after >= 0.0 && after <= 600.0))
        return CommandOutput::fail(fail("command.invalid_args", "kill_after_s is 0 to 600 seconds.", "args.kill_after_s"));
    RunEntry entry;
    if (Error e = runner.stop(run, after, &entry)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"run", run_entry_json(entry)}});
}

CommandOutput logs(const json& args, ModelRunner& runner) {
    const std::string run = str_arg(args, "run");
    if (run == kUserRunId)
        return CommandOutput::fail(fail(kRunNotManaged,
                                        "The run user is the person's own script, and the runner keeps the output of "
                                        "managed runs only.",
                                        "args.run"));
    int lines = 50;
    if (const auto it = args.find("lines"); it != args.end() && it->is_number_integer())
        lines = static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(200, it->get<std::int64_t>())));
    std::vector<std::string> out;
    if (Error e = runner.logs(run, lines, &out)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"run", run}, {"lines", out}});
}

}  // namespace

Error runner_precheck(const CommandInvocation& inv, ModelRunner* runner, const std::string& workspace) {
    if (!runner) return fail("command.not_available", "This session has no model runner.", inv.name);
    try {
        const json& args = inv.args;
        json params;
        std::string id;
        if (Error e = params_arg(args, &params)) return e;
        if (Error e = id_arg(args, &id)) return e;
        if (inv.name == "run.start") {
            RunStartRequest req;
            return script_arg(args, workspace, &req);
        }
        if (inv.name == "run.fork") {
            const std::string parent = str_arg(args, "run");
            if (parent == kUserRunId) return fail(kRunNotManaged, "The person's script cannot fork.", "args.run");
            for (const RunEntry& e : runner->list())
                if (e.id == parent && e.managed)
                    return e.state == "running" || e.state == "paused"
                               ? Error()
                               : fail(kRunNotReady, "The run " + parent + " is " + e.state + ".", "args.run");
            return fail(kRunNotFound, "There is no run " + parent + ".", "args.run");
        }
    } catch (const std::exception& e) {
        return fail(kRunStartFailed, e.what(), inv.name);
    }
    return {};
}

bool valid_run_id(const std::string& id) {
    if (id.empty() || id.size() > kRunMaxIdBytes) return false;
    for (const char ch : id)
        if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '-')) return false;
    return true;
}

std::string run_params_text(const json& params) {
    std::string out;
    auto add = [&out](const std::string& name, const json& value) {
        if (!out.empty()) out += ' ';
        out += name + "=" + value_text(value);
    };
    if (params.is_object()) {
        for (auto it = params.begin(); it != params.end(); ++it) add(it.key(), it.value());
    } else if (params.is_array()) {
        for (const auto& p : params)
            if (p.is_object() && p.contains("name") && p["name"].is_string() && p.contains("value"))
                add(p["name"].get<std::string>(), p["value"]);
    }
    return out;
}

std::string run_card_text(const CommandInvocation& inv, const std::string& script) {
    const json& a = inv.args;
    const auto params = [&a] {
        const auto it = a.find("params");
        return it == a.end() ? std::string() : cut_text(run_params_text(*it), kCardTextLimit);
    };
    if (inv.name == "run.start") {
        std::string args;
        if (const auto it = a.find("args"); it != a.end() && it->is_array())
            for (const json& x : *it)
                if (x.is_string()) args += (args.empty() ? "" : " ") + arg_text(x.get<std::string>());
        const std::string id = str_arg(a, "id");
        const std::string p = params();
        return "Start " + str_arg(a, "script") +
               (id.empty() ? std::string(" as a new managed run") : " as the managed run " + id) + ", with " +
               (args.empty() ? std::string("no arguments") : "the arguments " + cut_text(args, kCardTextLimit)) +
               " and " + (p.empty() ? std::string("no parameters") : "the parameters " + p) +
               ". The script runs as Python code on this computer.";
    }
    if (inv.name == "run.fork") {
        const std::string p = params();
        return "Fork the run " + str_arg(a, "run") + " at its next checkpoint, with " +
               (p.empty() ? std::string("no changed parameters") : "the parameters " + p) + ". The fork runs " +
               (script.empty() ? std::string("its script") : script) + " as Python code beside its parent.";
    }
    if (inv.name == "run.stop") {
        const auto k = a.find("kill_after_s");
        const double after = k != a.end() && k->is_number() ? k->get<double>() : 10.0;
        std::string secs = json(after).dump();
        if (secs.size() > 2 && secs.compare(secs.size() - 2, 2, ".0") == 0) secs.resize(secs.size() - 2);
        return "Stop the run " + str_arg(a, "run") + " at its next checkpoint, and end its process after " + secs +
               (secs == "1" ? " second." : " seconds.");
    }
    return {};
}

json run_entry_json(const RunEntry& e) {
    json out{{"id", e.id},           {"managed", e.managed},   {"state", e.state},       {"script", e.script},
             {"args", e.args},       {"attached", e.attached}, {"stopping", e.stopping}};
    json params = json::array();
    if (e.params.is_object())
        for (auto p = e.params.begin(); p != e.params.end(); ++p) params.push_back(json{{"name", p.key()}, {"value", p.value()}});
    out["params"] = std::move(params);
    out["parent"] = e.parent ? json(*e.parent) : json();
    out["step"] = e.step ? json(*e.step) : json();
    out["exit_code"] = e.exit_code ? json(*e.exit_code) : json();
    out["error"] = e.error ? json(*e.error) : json();
    return out;
}

CommandOutput runner_command(const CommandContext& context, ModelRunner* runner, const std::string& workspace,
                             bool approved) {
    const std::string& name = context.invocation.name;
    if (!runner)
        return CommandOutput::fail(fail("command.not_available",
                                        "Managed runs need the Caliper app, whose model runner starts them, and this "
                                        "session has no app.",
                                        name));
    try {
        const json& args = context.invocation.args;
        if (name == "run.list") {
            json runs = json::array();
            for (const auto& e : runner->list()) runs.push_back(run_entry_json(e));
            return CommandOutput::ok(json{{"runs", std::move(runs)}});
        }
        if (name == "run.start" || name == "run.fork") {
            // The same reasons as the policy's, before the approval counts.
            if (Error e = runner_precheck(context.invocation, runner, workspace)) return CommandOutput::fail(std::move(e));
            if (name == "run.start") return start(args, *runner, workspace, approved);
            return fork(args, *runner, approved);
        }
        if (name == "run.stop") return stop(args, *runner);
        if (name == "run.logs") return logs(args, *runner);
        return CommandOutput::fail(fail("command.unknown", "This is not a runner command.", name));
    } catch (const std::exception& e) {
        return CommandOutput::fail(fail(kRunStartFailed, std::string("The model runner failed: ") + e.what(), name));
    }
}

CommandOutput run_command(const CommandContext& context, RunControl* control, ModelRunner* runner) {
    const json& args = context.invocation.args;
    const auto it = args.find("run");
    if (it == args.end() || !it->is_string()) return run_command(context, control);
    const std::string run = it->get<std::string>();
    if (!runner) {
        if (run == kUserRunId) return run_command(context, control);
        return CommandOutput::fail(fail(kRunNotFound,
                                        "There is no run " + run + ": this session has no model runner, so it knows "
                                        "only the attached script, user.",
                                        "args.run"));
    }
    Error error;
    RunControl* target = runner->control_of(run, &error);
    if (!target) {
        if (!error) error = fail(kRunNotFound, "There is no run " + run + "; run.list lists the runs.", "args.run");
        return CommandOutput::fail(std::move(error));
    }
    return run_command(context, target);
}

}  // namespace caliper_builder
