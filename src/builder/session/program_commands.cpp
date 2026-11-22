// stage.check and program.status (design/gpu-programs.md, Commands), and the
// status item and the `program` event that the hosts share (program_host.h).
// The two commands answer command.not_implemented until their bodies come.
#include <caliper_builder/session/program_commands.h>

#include <string>
#include <utility>

namespace caliper_builder {

using nlohmann::json;

json program_status_json(const ProgramStatus& s) {
    auto opt = [](const auto& v) { return v ? json(*v) : json(nullptr); };
    json out = {{"id", s.id},
                {"kind", s.kind},
                {"host", s.host},
                {"state", s.state},
                {"every", opt(s.every)},
                {"every_min", opt(s.every_min)},
                {"last_pass_ms", opt(s.last_pass_ms)},
                {"last_cost_ms", opt(s.last_cost_ms)},
                {"step_ms", opt(s.step_ms)},
                {"last_step", opt(s.last_step)},
                {"runs", s.runs},
                {"memory_bytes", s.memory_bytes},
                {"memory_limit_bytes", s.memory_limit_bytes},
                {"error", nullptr},
                {"sha256", s.sha256}};
    if (s.error)
        out["error"] = {{"code", s.error->code},
                        {"message", s.error->message},
                        {"line", s.error->line},
                        {"column", s.error->column}};
    return out;
}

void push_program_event(EventSink& sink, const ProgramStatus& s, std::string_view change) {
    Event e;
    e.kind = EventKind::program;
    e.panel = s.panel;
    e.origin = s.host == "script" ? "script" : "ui";
    e.fields = {{"program", s.id}, {"change", std::string(change)}, {"host", s.host == "none" ? "viewer" : s.host}};
    if (s.error) {
        e.fields["code"] = s.error->code;
        e.fields["message"] = s.error->message;
        e.fields["line"] = s.error->line;
        e.fields["column"] = s.error->column;
    }
    sink.push(std::move(e));
}

CommandOutput program_command(const CommandContext& context, const spec::Spec&, const ProgramHost*,
                              const StageHost*) {
    return CommandOutput::fail(Error("command.not_implemented",
                                     context.invocation.name + " is not built yet (design/gpu-programs.md)."));
}

}  // namespace caliper_builder
