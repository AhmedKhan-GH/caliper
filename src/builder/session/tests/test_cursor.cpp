// The session tests of the cursor commands and the cursor events
// (design/instrument-interface.md, Cursor Commands and Cursor Events).
#include <doctest/doctest.h>

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/instrument_commands.h>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;

namespace {

struct Recorder : EventSink {
    std::vector<Event> events;
    std::uint64_t push(Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
};

TensorData one(float v) {
    TensorResult r = TensorData::allocate(DType::f32, {1});
    REQUIRE(r.error.ok());
    *reinterpret_cast<float*>(r.tensor.mutable_data()) = v;
    return r.tensor;
}

const char* kSpec =
    "caliper_spec = 1\n[applet]\nid = \"c\"\n"
    "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n";

struct Fixture {
    spec::Spec spec;
    ValueStore values;
    StepHistory history;
    Recorder events;
    Fixture() {
        spec::ParseResult r = spec::parse_toml(kSpec);
        REQUIRE(r.ok());
        spec = r.spec;
        values.reset(spec);
        history.configure(spec);
    }
    CommandResult run(const std::string& name, json args) {
        CommandInvocation inv;
        inv.name = name;
        inv.args = std::move(args);
        inv.origin = CommandOrigin::agent;
        CommandDescriptor d;
        d.name = name;
        CommandContext ctx{inv, d};
        InstrumentContext ic{spec, true, values, history, &events, nullptr, nullptr, {}};
        CommandOutput out = cursor_command(ctx, ic);
        CommandResult r;
        r.ok = out.error.ok();
        r.result = out.result;
        r.error = out.error;
        return r;
    }
};

}  // namespace

TEST_CASE("the cursor commands refuse a move with no kept steps") {
    Fixture f;
    CHECK(f.run("cursor.get", json::object()).ok);
    CHECK(f.run("cursor.set", {{"step", 3}}).error.code == "cursor.no_history");
    CHECK(f.run("cursor.step", {{"count", -1}}).error.code == "cursor.no_history");
    CHECK(f.run("cursor.step", {{"count", 0}}).error.code == "command.invalid_args");
    CHECK(f.run("cursor.set", {{"live", true}}).ok);
    CHECK(f.events.events.empty());
}

TEST_CASE("cursor.get gives the cursor and each source's kept steps") {
    Fixture f;
    for (int step : {5, 6, 7}) f.history.append("live:act", step, one(float(step)));
    const CommandResult r = f.run("cursor.get", json::object());
    REQUIRE(r.ok);
    CHECK(r.result["cursor"]["step"] == 7);
    CHECK(r.result["cursor"]["live"] == true);
    CHECK(r.result["cursor"]["first"] == 5);
    REQUIRE(r.result["sources"].size() == 1);
    CHECK(r.result["sources"][0] == json{{"id", "act"}, {"first", 5}, {"last", 7}, {"steps", 3},
                                         {"requested", 64}, {"bytes", 12}, {"note", nullptr}});
}

TEST_CASE("cursor.set and cursor.step move the cursor and add one cursor event each") {
    Fixture f;
    for (int step : {10, 20, 30}) f.history.append("live:act", step, one(float(step)));
    CommandResult r = f.run("cursor.set", {{"step", 18}});
    REQUIRE(r.ok);
    CHECK(r.result["cursor"]["step"] == 20);   // the nearest kept step
    CHECK(r.result["cursor"]["live"] == false);
    r = f.run("cursor.step", {{"count", -1}});
    CHECK(r.result["cursor"]["step"] == 10);
    r = f.run("cursor.set", {{"playing", true}, {"speed", 2.5}});
    CHECK(r.result["cursor"]["playing"] == true);
    CHECK(r.result["cursor"]["speed"] == 2.5);
    r = f.run("cursor.set", {{"playing", false}});
    r = f.run("cursor.set", {{"live", true}});
    CHECK(r.result["cursor"]["step"] == 30);
    CHECK(r.result["cursor"]["live"] == true);
    CHECK(f.run("cursor.set", {{"speed", 5000}}).error.code == "command.invalid_args");

    std::vector<std::string> changes;
    for (const Event& e : f.events.events) {
        CHECK(e.kind == EventKind::cursor);
        CHECK(e.origin == "agent");
        for (const char* key : {"change", "step", "first", "latest", "playing", "speed"}) CHECK(e.fields.contains(key));
        changes.push_back(e.fields["change"]);
    }
    CHECK(changes == std::vector<std::string>{"moved", "moved", "playing", "held", "live"});
    CHECK(f.events.events[0].fields["step"] == 20);
    // The value store follows the cursor for references to the state cursor.
    spec::Reference ref;
    ref.kind = spec::Reference::Kind::state;
    ref.target = "cursor";
    ref.field = "live";
    (void)f.values.value(ref);
}

TEST_CASE("the cursor commands pass the table's schemas through the session") {
    const std::string ws = fresh_dir("cursor-commands");
    auto s = open_session(ws);
    ok(*s, "spec.new", {{"id", "c"}});
    ok(*s, "spec.set_text", {{"text", std::string(kSpec)}});
    for (int step : {1, 2, 3, 4}) s->history().append("live:act", step, one(float(step)));
    json got = ok(*s, "cursor.get");
    CHECK(got["cursor"]["latest"] == 4);
    CHECK(got["sources"].size() == 1);
    got = ok(*s, "cursor.step", {{"count", -2}});
    CHECK(got["cursor"]["step"] == 2);
    got = ok(*s, "cursor.set", {{"live", true}});
    CHECK(got["cursor"]["step"] == 4);
    // Each command added its event to the session's ring.
    const EventRead read = s->events().read(0, EventFilter{{EventKind::cursor}, {}}, 16);
    CHECK(read.events.size() == 2);
}
