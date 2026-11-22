// The cursor commands cursor.get, cursor.set and cursor.step
// (design/instrument-interface.md, Commands, Cursor Commands and Cursor
// Events). They read and move the cursor of the session's StepHistory, copy
// it into the value store for references to the built-in state `cursor`,
// and add one `cursor` event for each change that a command makes.
#include <caliper_builder/session/instrument_commands.h>

namespace caliper_builder {

namespace {

using json = nlohmann::json;

Error no_history() {
    return Error("cursor.no_history",
                 "No source keeps steps yet, so the cursor has no step to move to; a source whose root is a live "
                 "tensor keeps 64 steps once its view sends values.",
                 "");
}

json sources_json(const StepHistory& history) {
    json out = json::array();
    for (const KeptSource& k : history.kept())
        out.push_back({{"id", k.id},
                       {"first", k.first},
                       {"last", k.last},
                       {"steps", k.steps},
                       {"requested", k.requested},
                       {"bytes", k.bytes},
                       {"note", k.note.empty() ? json(nullptr) : json(k.note)}});
    return out;
}

// Adds one `cursor` event (events.h) with the change and the cursor's
// fields.
void push_cursor_event(EventSink* events, const CursorState& c, const std::string& change, CommandOrigin origin) {
    if (!events) return;
    Event e;
    e.kind = EventKind::cursor;
    e.origin = command_origin_name(origin);
    e.fields = {{"change", change},          {"step", c.step},       {"first", c.first},
                {"latest", c.latest},        {"playing", c.playing}, {"speed", c.speed}};
    events->push(std::move(e));
}

}  // namespace

CommandOutput cursor_command(const CommandContext& ctx, InstrumentContext& ic) {
    const std::string& name = ctx.invocation.name;
    const json& args = ctx.invocation.args;
    StepHistory& history = ic.history;
    if (name == "cursor.get") {
        const CursorState c = history.cursor();
        ic.values.set_cursor(c);
        return CommandOutput::ok({{"cursor", cursor_json(c)}, {"sources", sources_json(history)}});
    }
    if (name == "cursor.step") {
        const json count = args.value("count", json());
        if (!count.is_number_integer() || count.get<std::int64_t>() == 0 || count.get<std::int64_t>() < -1000 ||
            count.get<std::int64_t>() > 1000)
            return CommandOutput::fail(
                Error("command.invalid_args", "count must be an integer from -1000 to 1000 and not 0.", "args.count"));
        if (history.kept_steps().empty()) return CommandOutput::fail(no_history());
        const CursorState c = history.step_cursor(count.get<std::int64_t>());
        ic.values.set_cursor(c);
        push_cursor_event(ic.events, c, "moved", ctx.invocation.origin);
        return CommandOutput::ok({{"cursor", cursor_json(c)}});
    }
    if (name != "cursor.set") return CommandOutput::fail(not_implemented(name));

    const CursorState before = history.cursor();
    CursorState wanted = before;
    const bool has_live = args.contains("live") && args["live"].is_boolean();
    const bool has_step = args.contains("step") && args["step"].is_number_integer();
    const bool has_playing = args.contains("playing") && args["playing"].is_boolean();
    if (args.contains("speed")) {
        const json& s = args["speed"];
        if (!s.is_number() || s.get<double>() < 0.1 || s.get<double>() > 1000.0)
            return CommandOutput::fail(
                Error("command.invalid_args", "speed must be a number from 0.1 to 1000.", "args.speed"));
        wanted.speed = s.get<double>();
    }
    if (has_step) wanted.step = args["step"].get<std::int64_t>();
    if (has_playing) wanted.playing = args["playing"].get<bool>();
    // A step or play holds the view unless live says otherwise; live = true
    // wins over both.
    if (has_step || (has_playing && wanted.playing)) wanted.live = false;
    if (has_live) wanted.live = args["live"].get<bool>();
    if (wanted.live) wanted.playing = false;
    const bool moves = has_step || has_playing || (has_live && !wanted.live);
    if (moves && !wanted.live && history.kept_steps().empty()) return CommandOutput::fail(no_history());
    const CursorState c = history.set_cursor(wanted);
    ic.values.set_cursor(c);
    std::string change;
    if (c.live && !before.live) change = "live";
    else if (c.playing && !before.playing) change = "playing";
    else if (has_step) change = "moved";
    else if (!c.live && before.live) change = "held";
    else if (!c.playing && before.playing) change = "held";
    if (!change.empty()) push_cursor_event(ic.events, c, change, ctx.invocation.origin);
    return CommandOutput::ok({{"cursor", cursor_json(c)}});
}

}  // namespace caliper_builder
