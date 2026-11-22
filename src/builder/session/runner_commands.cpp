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
#include <cstdint>
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

// Whether `s` is valid UTF-8 with no overlong form, surrogate or code point
// past U+10FFFF.
bool valid_utf8(const std::string& s) {
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t n = 0;
        std::uint32_t cp = 0;
        if (c < 0x80) {
            ++i;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            n = 1;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            n = 2;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            n = 3;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (i + n >= s.size()) return false;
        for (std::size_t k = 1; k <= n; ++k) {
            const unsigned char d = static_cast<unsigned char>(s[i + k]);
            if ((d & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3F);
        }
        const std::uint32_t least = n == 1 ? 0x80 : n == 2 ? 0x800 : 0x10000;
        if (cp < least || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += n + 1;
    }
    return true;
}

// Whether a code point changes how text around it reads without a glyph of
// its own: the C1 controls, the soft hyphen, the Arabic letter mark, the
// zero-width and direction marks, the line and paragraph separators, the
// direction embeddings, overrides and isolates, the invisible operators, the
// byte order mark, the interlinear annotation marks and the tag characters.
bool format_code_point(std::uint32_t cp) {
    return (cp >= 0x80 && cp <= 0x9F) || cp == 0xAD || cp == 0x61C || (cp >= 0x200B && cp <= 0x200F) ||
           (cp >= 0x2028 && cp <= 0x202E) || (cp >= 0x2060 && cp <= 0x206F) || cp == 0xFEFF ||
           (cp >= 0xFFF9 && cp <= 0xFFFB) || (cp >= 0xE0000 && cp <= 0xE007F);
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

// A text as a card shows it (run_card_text()): bare when it is a plain word of
// printable ASCII, else in double quotes with JSON's escapes, and each
// control or format character as \uXXXX, so that no text reads as two or
// ends the card. `name` also quotes a word with '=', and `value` a word that
// reads as a number, true or false.
std::string shown(const std::string& s, bool name = false, bool value = false) {
    bool plain = !s.empty() && !(value && reads_as_scalar(s));
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c <= 0x20 || c >= 0x7F || c == '"' || c == '\\' || c == ',' || (name && c == '=')) plain = false;
    }
    if (plain) return s;
    static const char* hex = "0123456789abcdef";
    auto u = [](std::string& out, std::uint32_t cp) {
        out += "\\u";
        for (int shift = 12; shift >= 0; shift -= 4) out += hex[(cp >> shift) & 0xF];
    };
    std::string out = "\"";
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n') out += "\\n";
            else if (c == '\r') out += "\\r";
            else if (c == '\t') out += "\\t";
            else if (c < 0x20 || c == 0x7F) u(out, c);
            else out += static_cast<char>(c);
            ++i;
            continue;
        }
        const std::size_t n = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        const std::string one = s.substr(i, n);
        if (n == 1 || !valid_utf8(one)) {   // the handlers refuse it; shown byte by byte all the same
            out += "\\x";
            out += hex[c >> 4];
            out += hex[c & 0xF];
            ++i;
            continue;
        }
        std::uint32_t cp = c & (n == 2 ? 0x1F : n == 3 ? 0x0F : 0x07);
        for (std::size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        if (!format_code_point(cp)) {
            out += one;
        } else if (cp <= 0xFFFF) {
            u(out, cp);
        } else {   // a surrogate pair, as JSON writes it
            u(out, 0xD800 + ((cp - 0x10000) >> 10));
            u(out, 0xDC00 + ((cp - 0x10000) & 0x3FF));
        }
        i += n;
    }
    return out + "\"";
}

// A value as a card shows it: a string as shown() gives it, a Boolean as true
// or false, and a number as JSON writes it.
std::string card_value(const json& v) {
    if (v.is_string()) return shown(v.get<std::string>(), false, true);
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    return v.dump();
}

// A parameter name as caliper.param takes it: 1 to 63 bytes, no control
// characters.
bool valid_param_name(const std::string& name) {
    if (name.empty() || name.size() > 63) return false;
    for (const unsigned char ch : name)
        if (ch < 0x20 || ch == 0x7f) return false;
    return true;
}

// The `params` argument, a list of {name, value}, as an object of values by
// name, or an Error. The schema checked the count, the names and the types;
// the byte limits are checked here, so that the card shows every value in
// full.
Error params_arg(const json& args, json* out) {
    *out = json::object();
    const auto it = args.find("params");
    if (it == args.end()) return {};
    if (!it->is_array() || it->size() > kRunMaxParams)
        return fail("command.invalid_args", "params is a list of at most 64 {name, value} pairs.", "args.params");
    std::size_t total = 0;
    for (const auto& p : *it) {
        const std::string name = p.is_object() ? str_arg(p, "name") : std::string();
        if (!valid_param_name(name) || !valid_utf8(name))
            return fail("command.invalid_args",
                        "A parameter name has 1 to 63 bytes of UTF-8 and no control characters, as caliper.param "
                        "takes it.",
                        "args.params");
        const json v = p.value("value", json());
        if (!(v.is_number() || v.is_boolean() || v.is_string()))
            return fail("command.invalid_args",
                        "The parameter " + shown(name, true) + " takes a number, true or false, or a string.",
                        "args.params");
        total += name.size();
        if (v.is_string()) {
            const std::string& text = v.get_ref<const std::string&>();
            if (text.size() > kRunMaxParamValueBytes || !valid_utf8(text))
                return fail("command.invalid_args",
                            "The string value of the parameter " + shown(name, true) + " has at most " +
                                std::to_string(kRunMaxParamValueBytes) + " bytes of UTF-8.",
                            "args.params");
            total += text.size();
        }
        if (total > kRunMaxParamsBytes)
            return fail("command.invalid_args",
                        "The parameter names and string values have at most " + std::to_string(kRunMaxParamsBytes) +
                            " bytes in all, so that the card shows them in full.",
                        "args.params");
        if (out->contains(name))
            return fail("command.invalid_args", "The parameter " + shown(name, true) + " is named twice.",
                        "args.params");
        (*out)[name] = v;
    }
    return {};
}

// The `args` argument as strings, or an Error: at most kRunMaxArgs, each at
// most kRunMaxArgBytes of UTF-8 and kRunMaxArgsBytes in all, so that the card
// shows every argument in full.
Error args_arg(const json& args, std::vector<std::string>* out) {
    out->clear();
    const auto it = args.find("args");
    if (it == args.end()) return {};
    if (!it->is_array() || it->size() > kRunMaxArgs)
        return fail("command.invalid_args", "args is a list of at most 64 strings.", "args.args");
    std::size_t total = 0;
    for (const auto& a : *it) {
        if (!a.is_string()) return fail("command.invalid_args", "Each argument is a string.", "args.args");
        const std::string& text = a.get_ref<const std::string&>();
        if (text.size() > kRunMaxArgBytes || !valid_utf8(text))
            return fail("command.invalid_args",
                        "An argument has at most " + std::to_string(kRunMaxArgBytes) + " bytes of UTF-8.", "args.args");
        total += text.size();
        if (total > kRunMaxArgsBytes)
            return fail("command.invalid_args",
                        "The arguments have at most " + std::to_string(kRunMaxArgsBytes) +
                            " bytes in all, so that the card shows them in full.",
                        "args.args");
        out->push_back(text);
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

// The script argument as a canonical file in the workspace. The argument
// must be that file's path relative to the workspace, with '/' and no '.',
// '..' or link in it, so that the card, which shows the argument, names the
// file that runs. The handler checks it again when the start runs, so a link
// made after the card was approved refuses the start.
Error script_arg(const json& args, const std::string& workspace, RunStartRequest* out) {
    const std::string script = str_arg(args, "script");
    const std::string refused = "The script " + shown(script) + " is not a file in the workspace " + workspace +
                                "; a managed run starts only a script of the workspace.";
    if (script.empty() || !valid_utf8(script)) return fail(kRunScriptRefused, refused, "args.script");
    std::error_code ec;
    fs::path p = fs::u8path(script);
    if (p.is_relative()) p = fs::u8path(workspace) / p;
    const fs::path canon = fs::canonical(p, ec);
    if (ec) return fail(kRunScriptRefused, refused, "args.script");
    const std::string canon_text = canon.u8string();
    if (!path_within(canon_text, workspace) || !fs::is_regular_file(canon, ec) || ec)
        return fail(kRunScriptRefused, refused, "args.script");
    const std::string relative = canon.lexically_relative(fs::u8path(workspace)).generic_u8string();
    if (script != relative)
        return fail(kRunScriptRefused,
                    "The script " + shown(script) + " is the file " + shown(relative) +
                        " of the workspace; give that path, relative to the workspace and with no '.', '..' or link, "
                        "so that the card names the file that runs.",
                    "args.script");
    out->script_path = canon_text;
    out->script = relative;
    return {};
}

CommandOutput start(const json& args, ModelRunner& runner, const std::string& workspace, bool approved) {
    RunStartRequest req;
    if (Error e = script_arg(args, workspace, &req)) return CommandOutput::fail(std::move(e));
    if (Error e = args_arg(args, &req.args)) return CommandOutput::fail(std::move(e));
    if (Error e = params_arg(args, &req.params)) return CommandOutput::fail(std::move(e));
    if (Error e = id_arg(args, &req.id)) return CommandOutput::fail(std::move(e));
    if (!approved) return CommandOutput::fail(not_approved("run.start"));
    RunEntry entry;
    if (Error e = runner.start(req, &entry)) return CommandOutput::fail(std::move(e));
    return CommandOutput::ok(json{{"run", run_entry_json(entry)}});
}

// A step of JSON: a non-negative integer, signed or unsigned, else nothing.
std::optional<std::uint64_t> step_value(const json& v) {
    if (v.is_number_unsigned()) return v.get<std::uint64_t>();
    if (v.is_number_integer() && v.get<std::int64_t>() >= 0) return static_cast<std::uint64_t>(v.get<std::int64_t>());
    return std::nullopt;
}

// The step and the interventions of a fork: each intervention an object with
// a point, the op ablate and optional steps, with until_step above
// from_step, at most kRunMaxForkInterventions of them, and their points
// together at most kRunMaxInterventionsBytes. A point that the parent does
// not list is intervene.unknown_point when `runner` gives the parent's points;
// the fork's script refuses it at its start otherwise.
Error fork_args(const json& args, ModelRunner* runner, RunForkRequest* out) {
    if (const auto it = args.find("step"); it != args.end()) {
        out->step = step_value(*it);
        if (!out->step) return fail("command.invalid_args", "step is a step of the run, 0 or more.", "args.step");
    }
    out->interventions = json::array();
    const auto it = args.find("interventions");
    if (it == args.end()) return {};
    if (!it->is_array() || it->size() > kRunMaxForkInterventions)
        return fail("command.invalid_args",
                    "interventions is an array of at most " + std::to_string(kRunMaxForkInterventions) + " interventions.",
                    "args.interventions");
    const std::vector<InterventionPoint> points =
        runner ? runner->intervention_points(str_arg(args, "run")) : std::vector<InterventionPoint>();
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < it->size(); ++i) {
        const json& iv = (*it)[i];
        const std::string where = "args.interventions[" + std::to_string(i) + "]";
        if (!iv.is_object()) return fail("command.invalid_args", "An intervention is an object.", where);
        const std::string point = str_arg(iv, "point");
        const std::string op = str_arg(iv, "op");
        if (point.empty() || !valid_utf8(point))
            return fail("command.invalid_args", "An intervention names its point.", where + ".point");
        bytes += point.size();
        if (bytes > kRunMaxInterventionsBytes)
            return fail("command.invalid_args",
                        "The points of the interventions hold at most " + std::to_string(kRunMaxInterventionsBytes) +
                            " bytes together.",
                        where + ".point");
        if (op != "ablate")
            return fail(kInterveneOpReserved, "The op " + op + " comes later; ablate is the only op built.", where + ".op");
        std::optional<std::uint64_t> steps[2];
        const char* keys[2] = {"from_step", "until_step"};
        for (int k = 0; k < 2; ++k)
            if (const auto s = iv.find(keys[k]); s != iv.end()) {
                steps[k] = step_value(*s);
                if (!steps[k]) return fail("command.invalid_args", std::string(keys[k]) + " is a step.", where + "." + keys[k]);
            }
        if (steps[0] && steps[1] && *steps[1] <= *steps[0])
            return fail(kInterveneBadRange,
                        "until_step " + std::to_string(*steps[1]) + " is not above from_step " + std::to_string(*steps[0]) + ".",
                        where + ".until_step");
        if (!points.empty() &&
            std::none_of(points.begin(), points.end(), [&](const InterventionPoint& p) { return p.point == point; }))
            return fail(kInterveneUnknownPoint,
                        "The run " + str_arg(args, "run") + " has no point " + point + "; intervene.points lists them.",
                        where + ".point");
        json one = {{"point", point}, {"op", op}};
        if (steps[0]) one["from_step"] = *steps[0];
        if (steps[1]) one["until_step"] = *steps[1];
        out->interventions.push_back(std::move(one));
    }
    return {};
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
    if (Error e = fork_args(args, &runner, &req)) return CommandOutput::fail(std::move(e));
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
            if (Error e = args_arg(args, &req.args)) return e;
            return script_arg(args, workspace, &req);
        }
        if (inv.name == "run.fork") {
            const std::string parent = str_arg(args, "run");
            if (parent == kUserRunId) return fail(kRunNotManaged, "The person's script cannot fork.", "args.run");
            RunForkRequest req;
            if (Error e = fork_args(args, runner, &req)) return e;
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
        out += shown(name, true) + "=" + card_value(value);
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
    // Every text in full and escaped (shown()): the handlers refuse what
    // passes the limits, so nothing here is cut.
    const json& a = inv.args;
    const auto params = [&a] {
        const auto it = a.find("params");
        return it != a.end() && it->is_array() ? run_params_text(*it) : std::string();
    };
    if (inv.name == "run.start") {
        std::string args;
        if (const auto it = a.find("args"); it != a.end() && it->is_array())
            for (const json& x : *it)
                if (x.is_string()) args += (args.empty() ? "" : " ") + shown(x.get<std::string>());
        const std::string id = str_arg(a, "id");
        const std::string p = params();
        return "Start " + shown(str_arg(a, "script")) +
               (id.empty() ? std::string(" as a new managed run") : " as the managed run " + shown(id)) + ", with " +
               (args.empty() ? std::string("no arguments") : "the arguments " + args) + " and " +
               (p.empty() ? std::string("no parameters") : "the parameters " + p) +
               ". The script runs as Python code on this computer.";
    }
    if (inv.name == "run.fork") {
        const std::string p = params();
        const auto st = a.find("step");
        const std::optional<std::uint64_t> step = st != a.end() ? step_value(*st) : std::nullopt;
        const std::string from = step ? " from its newest kept checkpoint at or before step " + std::to_string(*step)
                                      : std::string(" at its next checkpoint");
        std::string ivs;
        if (const auto it = a.find("interventions"); it != a.end() && it->is_array())
            for (const json& iv : *it) {
                if (!iv.is_object()) continue;
                ivs += (ivs.empty() ? "" : ", ") + shown(str_arg(iv, "op")) + " " + shown(str_arg(iv, "point"));
                const auto f = iv.find("from_step");
                const auto u = iv.find("until_step");
                const std::optional<std::uint64_t> from_step = f != iv.end() ? step_value(*f) : std::nullopt;
                const std::optional<std::uint64_t> until_step = u != iv.end() ? step_value(*u) : std::nullopt;
                ivs += from_step ? " from step " + std::to_string(*from_step) : std::string(" from its first step");
                if (until_step) ivs += " until step " + std::to_string(*until_step);
            }
        return "Fork the run " + shown(str_arg(a, "run")) + from + ", with " +
               (p.empty() ? std::string("no changed parameters") : "the parameters " + p) +
               (ivs.empty() ? std::string() : " and the interventions " + ivs) + ". The fork runs " +
               (script.empty() ? std::string("its script") : shown(script)) + " as Python code beside its parent.";
    }
    if (inv.name == "run.stop") {
        const auto k = a.find("kill_after_s");
        const double after = k != a.end() && k->is_number() ? k->get<double>() : 10.0;
        std::string secs = json(after).dump();
        if (secs.size() > 2 && secs.compare(secs.size() - 2, 2, ".0") == 0) secs.resize(secs.size() - 2);
        return "Stop the run " + shown(str_arg(a, "run")) + " at its next checkpoint, and end its process after " +
               secs + (secs == "1" ? " second." : " seconds.");
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
