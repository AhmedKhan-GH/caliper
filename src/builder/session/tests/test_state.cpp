// The session tests of the state commands, control.set, the selection commands and the state events
// (design/instrument-interface.md, Shared State and Commands).
#include <doctest/doctest.h>

#include <caliper_builder/session/instrument_commands.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/session/values.h>

#include "helpers.h"
#include "../../../../tests/gfx/example_pieces.h"   // the attention arcs piece and its resources

#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/programs/pieces.h>

#include <limits>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

// A session with the spec "bench", the state token (integer 0 to 63) and the state measure (norm or mean).
std::unique_ptr<Session> bench(const std::string& name) {
    auto s = open_session(fresh_dir(name) + "/workspace");
    ok(*s, "spec.new", {{"id", "bench"}});
    ok(*s, "state.create", {{"id", "token"}, {"type", "integer"}, {"min", 0}, {"max", 63}, {"default", 0}});
    ok(*s, "state.create", {{"id", "measure"}, {"type", "text"}, {"choices", {"norm", "mean"}}, {"default", "norm"}});
    return s;
}

void add_pick(Session& s) {
    ok(s, "control.create",
       {{"id", "pick"},
        {"kind", "segmented"},
        {"bind", "state:measure"},
        {"options", {{"choices", {{{"value", "norm"}, {"label", "Norm"}, {"key", "n"}},
                                  {{"value", "mean"}, {"label", "Mean"}, {"key", "m"}}}}}}});
}

std::vector<Event> events_of(Session& s, EventKind kind) {
    return s.events().read(0, EventFilter{{kind}, {}}, 100).events;
}

// Keeps the events that a ValueStore pushes.
struct Recorder final : EventSink {
    std::vector<Event> events;
    std::uint64_t push(Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
};

spec::Spec parse(const std::string& body, const std::string& applet = "a") {
    auto parsed = spec::parse_toml("caliper_spec = 1\n[applet]\nid = \"" + applet + "\"\n" + body);
    REQUIRE_MESSAGE(parsed.ok(), body);
    return parsed.spec;
}

spec::Reference control_ref(const std::string& id, const char* field = nullptr) {
    spec::Reference r;
    r.kind = spec::Reference::Kind::control;
    r.target = id;
    if (field) r.field = field;
    return r;
}

spec::Reference state_ref(const std::string& id) {
    spec::Reference r;
    r.kind = spec::Reference::Kind::state;
    r.target = id;
    return r;
}

}  // namespace

// state.create, state.patch and the refusals of both.
TEST_CASE("state: create and patch") {
    auto s = bench("state-create");
    CHECK(err(*s, "state.create", {{"id", "token"}, {"type", "integer"}}) == "state.exists");
    CHECK(err(*s, "state.patch", {{"id", "nope"}, {"set", {{"max", 5}}}}) == "state.not_found");
    CHECK(err(*s, "state.create", {{"id", "big"}, {"type", "integer"}, {"min", 0}, {"max", 63}, {"default", 99}}) ==
          "spec.out_of_range");
    ok(*s, "state.patch", {{"id", "token"}, {"set", {{"max", 127}}}});
    ok(*s, "state.set", {{"id", "token"}, {"value", 100}});
}

// state.set answers with the value and the previous one, and pushes one state event with the caller's origin.
TEST_CASE("state: set and its event") {
    auto s = bench("state-set");
    json r = ok(*s, "state.set", {{"id", "token"}, {"value", 5}}, CommandOrigin::agent);
    CHECK(r["value"] == 5);
    CHECK(r["previous"] == 0);
    auto events = events_of(*s, EventKind::state);
    REQUIRE(events.size() == 1);
    CHECK(events[0].fields["state"] == "token");
    CHECK(events[0].fields["value"] == 5);
    CHECK(events[0].fields["previous"] == 0);
    CHECK(events[0].fields["writer"] == "state.set");
    CHECK(events[0].origin == "agent");

    ok(*s, "state.set", {{"id", "token"}, {"value", 5}});
    CHECK(events_of(*s, EventKind::state).size() == 1);

    CHECK(err(*s, "state.set", {{"id", "token"}, {"value", 99}}) == "state.bad_value");
    CHECK(err(*s, "state.set", {{"id", "token"}, {"value", "x"}}) == "state.bad_value");
    CHECK(err(*s, "state.set", {{"id", "cursor"}, {"value", 1}}) == "state.read_only");
    CHECK(err(*s, "state.set", {{"id", "nope"}, {"value", 1}}) == "state.not_found");
    CHECK(events_of(*s, EventKind::state).size() == 1);
}

// state.get lists every state with its fields, or only the ids asked for.
TEST_CASE("state: get") {
    auto s = bench("state-get");
    json all = ok(*s, "state.get")["states"];
    REQUIRE(all.size() == 2);
    for (const char* key : {"id", "type", "value", "default", "writers", "readers"}) CHECK(all[0].contains(key));
    CHECK(all[0]["id"] == "token");
    CHECK(all[0]["type"] == "integer");
    CHECK(all[1]["value"] == "norm");

    json one = ok(*s, "state.get", {{"ids", {"measure"}}})["states"];
    REQUIRE(one.size() == 1);
    CHECK(one[0]["id"] == "measure");
    CHECK(err(*s, "state.get", {{"ids", {"nope"}}}) == "state.not_found");
}

// control.set on a segmented group writes its state, with a control event and a state event.
TEST_CASE("state: control.set of a segmented group") {
    auto s = bench("state-control-segmented");
    add_pick(*s);
    json r = ok(*s, "control.set", {{"id", "pick"}, {"value", "mean"}});
    CHECK(r["value"] == "mean");

    auto controls = events_of(*s, EventKind::control);
    REQUIRE(controls.size() == 1);
    CHECK(controls[0].fields["control"] == "pick");
    CHECK(controls[0].fields["bind"] == "state:measure");
    CHECK(controls[0].fields["value"] == "mean");
    auto states = events_of(*s, EventKind::state);
    REQUIRE(states.size() == 1);
    CHECK(states[0].fields["state"] == "measure");
    CHECK(states[0].fields["writer"] == "pick");

    json measure = ok(*s, "state.get", {{"ids", {"measure"}}})["states"][0];
    CHECK(measure["value"] == "mean");
    CHECK(measure["writers"] == json::array({"control[0]"}));

    CHECK(err(*s, "control.set", {{"id", "pick"}, {"value", "other"}}) == "state.bad_value");
    CHECK(err(*s, "control.set", {{"id", "nope"}, {"value", 1}}) == "control.not_found");
}

// A range control widens to min_span and clamps to its bounds; a parameter control cannot be set.
TEST_CASE("state: control.set of a range and a parameter slider") {
    auto s = bench("state-control-range");
    ok(*s, "control.create",
       {{"id", "win"}, {"kind", "range"}, {"bind", "value"},
        {"options", {{"min", 0}, {"max", 100}, {"value", {0, 50}}, {"min_span", 10}}}});
    json widened = ok(*s, "control.set", {{"id", "win"}, {"value", {40, 42}}})["value"];
    CHECK(widened == json::array({40, 50}));
    json clamped = ok(*s, "control.set", {{"id", "win"}, {"value", {90, 200}}})["value"];
    REQUIRE(clamped.size() == 2);
    CHECK(clamped[0].get<double>() >= 0);
    CHECK(clamped[1].get<double>() <= 100);
    CHECK(clamped[1].get<double>() - clamped[0].get<double>() >= 10);

    ok(*s, "control.create", {{"id", "lr"}, {"kind", "slider"}, {"bind", "param:lr"}, {"options", {{"min", 0}, {"max", 1}}}});
    CHECK(err(*s, "control.set", {{"id", "lr"}, {"value", 0.5}}) == "control.not_settable");
}

// A state that a control writes cannot be removed until the control goes.
TEST_CASE("state: remove while a control binds it") {
    auto s = bench("state-remove");
    add_pick(*s);
    CHECK(err(*s, "state.remove", {{"id", "measure"}}) == "state.in_use");
    ok(*s, "control.remove", {{"id", "pick"}});
    ok(*s, "state.remove", {{"id", "measure"}});
}

// Each control kind fits a written value to its own rules, and a reference reads the result.
TEST_CASE("state: control kinds write through the value store") {
    const spec::Spec sp = parse(R"(
[[control]]
id = "slide"
kind = "slider"
bind = "value"
options = { min = 0, max = 10, step = 2 }

[[control]]
id = "flag"
kind = "toggle"
bind = "value"

[[control]]
id = "choose"
kind = "picker"
bind = "value"
options = { choices = ["a", "b"] }

[[control]]
id = "name"
kind = "text"
bind = "value"
options = { max_length = 5 }

[[control]]
id = "many"
kind = "segmented"
bind = "value"
options = { multiple = true, choices = ["a", "b", "c"] }

[[control]]
id = "span"
kind = "range"
bind = "value"
options = { min = 0, max = 10, value = [2, 4] }
)");
    ValueStore values;
    values.reset(sp);
    auto set = [&](const char* id, json v) { return values.set_control(id, v, CommandOrigin::ui, nullptr); };

    CHECK(set("slide", 5.1).ok());
    CHECK(values.control_value("slide") == 6);
    CHECK(set("slide", 100).ok());
    CHECK(values.control_value("slide") == 10);
    CHECK(set("slide", "x").code == "state.bad_value");

    CHECK(set("flag", 1).code == "state.bad_value");
    CHECK(set("flag", true).ok());
    CHECK(values.control_value("flag") == true);

    CHECK(set("choose", "c").code == "state.bad_value");
    CHECK(set("choose", "b").ok());
    CHECK(values.control_value("choose") == "b");

    CHECK(set("name", "abcdef").code == "state.bad_value");
    CHECK(set("name", "abc").ok());
    CHECK(values.control_value("name") == "abc");

    CHECK(set("many", json::array({"c", "a"})).ok());
    CHECK(values.control_value("many") == json::array({"a", "c"}));

    CHECK(set("span", json::array({6, 8})).ok());
    CHECK(values.value(control_ref("slide")) == 10);
    CHECK(values.value(control_ref("choose")) == "b");
    CHECK(values.value(control_ref("many")) == json::array({"a", "c"}));
    CHECK(values.value(control_ref("span", "lo")) == 6);
    CHECK(values.value(control_ref("span", "hi")) == 8);
}

// A pick on the entity of a select option writes the state, and a selection reference reads the pick.
TEST_CASE("state: picks write the select state") {
    const spec::Spec sp = parse(R"(
[[state]]
id = "token"
type = "integer"
min = 0
max = 63

[[panel]]
id = "world"
primitive = "scene"
bind = { scene = "live:model" }
options = { select = { state = "token", field = "column", entity = "terrain" } }
)");
    ValueStore values;
    values.reset(sp);
    Recorder recorder;

    PickFields terrain;
    terrain.entity = "terrain";
    terrain.column = 7;
    values.record_pick("world", terrain, CommandOrigin::ui, &recorder);
    CHECK(values.value(state_ref("token")) == 7);
    REQUIRE(recorder.events.size() == 1);
    CHECK(recorder.events[0].kind == EventKind::state);
    CHECK(recorder.events[0].fields["writer"] == "world");
    CHECK(recorder.events[0].fields["value"] == 7);

    spec::Reference column;
    column.kind = spec::Reference::Kind::selection;
    column.target = "world";
    column.field = "column";
    column.fallback = -1;
    REQUIRE(values.last_pick("world").has_value());
    CHECK(values.last_pick("world")->column == std::optional<std::int64_t>(7));
    CHECK(values.value(column) == 7);

    // A pick on another entity writes nothing, but it is the last pick.
    PickFields other;
    other.entity = "other";
    other.column = 3;
    values.record_pick("world", other, CommandOrigin::ui, &recorder);
    CHECK(recorder.events.size() == 1);
    CHECK(values.value(state_ref("token")) == 7);
    CHECK(values.value(column) == 3);

    values.clear_pick("world");
    CHECK_FALSE(values.last_pick("world").has_value());
    CHECK(values.value(column) == -1);
}

// A program's pick gives its item map's own field when the field's name is also a key of the pick, such as
// index: a select and a selection reference read the item's value, not the pick's tensor indices.
TEST_CASE("state: an item map field named as a pick key reads the item's value") {
    const spec::Spec sp = parse(R"(
[[state]]
id = "point"
type = "integer"
min = 0
max = 1000

[[panel]]
id = "world"
primitive = "scene"
bind = { scene = "live:model" }
options = { select = { state = "point", field = "index" } }
)");
    ValueStore values;
    values.reset(sp);
    Recorder recorder;
    PickFields picked;
    picked.entity = "cloud";
    picked.program = "cloud";
    picked.item = 41;
    picked.step = 9;
    picked.program_fields = {{"index", 41.0}};
    picked.tensor_index = std::array<std::int64_t, 4>{41, -1, -1, -1};
    values.record_pick("world", picked, CommandOrigin::ui, &recorder);
    CHECK(values.value(state_ref("point")) == 41);
    spec::Reference index;
    index.kind = spec::Reference::Kind::selection;
    index.target = "world";
    index.field = "index";
    CHECK(values.value(index) == 41.0);
    // The pick's own keys stay as they were.
    CHECK(pick_fields_json(picked)["index"] == json::array({41, -1, -1, -1}));
}

// selection.get and selection.set through the session.
TEST_CASE("state: selection commands") {
    auto s = open_session(fresh_dir("state-selection") + "/workspace");
    ok(*s, "spec.new", {{"id", "bench"}});
    ok(*s, "panel.create", {{"id", "world"}, {"primitive", "scene"}, {"bind", {{"scene", "live:model"}}}});
    ok(*s, "source.add", {{"id", "src"}, {"catalog", "live:x"}});
    ok(*s, "panel.create", {{"id", "heat"}, {"primitive", "heatmap"}, {"bind", {{"values", "src"}}}});

    CHECK(ok(*s, "selection.get", {{"panel", "world"}})["fields"].is_null());
    json cleared = ok(*s, "selection.set", {{"panel", "world"}, {"clear", true}});
    CHECK(cleared["fields"].is_null());
    auto selects = events_of(*s, EventKind::select);
    REQUIRE(selects.size() == 1);
    CHECK(selects[0].fields["entities"] == json::array());

    CHECK(err(*s, "selection.set", {{"panel", "nope"}, {"clear", true}}) == "panel.not_found");
    CHECK(err(*s, "selection.set", {{"panel", "heat"}, {"entity", "e"}, {"index", 0}}) == "selection.bad_item");
}

// selection.set reads an item only inside the entity's arrays: a negative
// index and one that wraps to a negative one are refused at the command, and
// one past the items at the data.
TEST_CASE("state: selection.set refuses an index outside the entity's items") {
    const std::string ws = fresh_dir("state-selection-range") + "/workspace";
    write_safetensors_f32(ws + "/data/cloud.safetensors", "p", {4, 3}, {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3});
    write_safetensors_f32(ws + "/data/hills.safetensors", "h", {2, 3}, {1, 2, 3, 4, 5, 6});
    auto s = open_session(ws);
    ok(*s, "data.import", {{"paths", {"data/cloud.safetensors", "data/hills.safetensors"}}});
    ok(*s, "spec.new", {{"id", "bench"}});
    ok(*s, "source.add", {{"id", "cloud"}, {"catalog", "tensor:cloud_p"}});
    ok(*s, "source.add", {{"id", "hills"}, {"catalog", "tensor:hills_h"}});
    ok(*s, "panel.create", {{"id", "world"}, {"primitive", "scene"}, {"bind", {{"scene", "live:model"}}}});
    ok(*s, "entity.set", {{"panel", "world"}, {"entity", {{"path", "dots"}, {"draw", "points"}, {"bind", {{"positions", "cloud"}}}}}});
    ok(*s, "entity.set", {{"panel", "world"}, {"entity", {{"path", "land"}, {"draw", "surface"}, {"bind", {{"height", "hills"}}}}}});

    const json last = ok(*s, "selection.set", {{"panel", "world"}, {"entity", "dots"}, {"index", 3}});
    CHECK(last["fields"]["index"] == 3);
    CHECK(last["fields"]["x"] == 3.0);
    const json cell = ok(*s, "selection.set", {{"panel", "world"}, {"entity", "land"}, {"index", 5}});
    CHECK(cell["fields"]["z"] == 6.0);
    for (const char* entity : {"dots", "land"}) {
        INFO(entity);
        // The command's schema refuses a negative value (command.invalid_args);
        // an unsigned value past the signed range passes it and would wrap
        // to -1, which the command refuses.
        for (const json& index : {json(-1), json(std::numeric_limits<std::int64_t>::min()),
                                  json(std::numeric_limits<std::uint64_t>::max())}) {
            INFO(index.dump());
            const auto r = invoke(*s, "selection.set", {{"panel", "world"}, {"entity", entity}, {"index", index}});
            CHECK_FALSE(r.ok);
            CHECK(r.error.where == "args.index");
            if (index.is_number_unsigned()) CHECK(r.error.code == "selection.bad_item");
        }
        CHECK(err(*s, "selection.set", {{"panel", "world"}, {"entity", entity}, {"index", 6}}) == "selection.bad_item");
    }
    // The refused calls left the last pick as it was.
    CHECK(ok(*s, "selection.get", {{"panel", "world"}})["fields"]["entity"] == "land");
}

// A range with follow moves with its upper bound while its upper end stands on it.
TEST_CASE("state: follow") {
    const spec::Spec sp = parse(R"(
[[control]]
id = "window"
kind = "range"
bind = "value"
options = { min = { state = "cursor", field = "first" }, max = { state = "cursor", field = "latest" }, value = [0, 2048], follow = true, min_span = 16 }
)");
    ValueStore values;
    values.reset(sp);
    auto cursor = [&](std::int64_t first, std::int64_t latest) {
        CursorState c;
        c.first = first;
        c.latest = latest;
        values.set_cursor(c);
    };
    cursor(0, 30);
    CHECK(values.control_value("window") == json::array({0, 30}));
    cursor(0, 3000);
    CHECK(values.control_value("window") == json::array({952, 3000}));

    // A window moved off the newest step stays.
    REQUIRE(values.set_control("window", json::array({100, 200}), CommandOrigin::ui, nullptr).ok());
    cursor(0, 3100);
    CHECK(values.control_value("window") == json::array({100, 200}));

    // A window put on the newest step follows again.
    REQUIRE(values.set_control("window", json::array({3000, 3100}), CommandOrigin::ui, nullptr).ok());
    cursor(0, 3200);
    CHECK(values.control_value("window") == json::array({3100, 3200}));
}

// A reset for the same applet keeps a state's value, and one for another applet starts at the default.
TEST_CASE("state: reset") {
    const std::string body = "[[state]]\nid = \"token\"\ntype = \"integer\"\nmin = 0\nmax = 63\n";
    ValueStore values;
    values.reset(parse(body, "a"));
    REQUIRE(values.set_state("token", 5, "test", CommandOrigin::ui, nullptr).ok());
    values.reset(parse(body, "a"));
    CHECK(values.value(state_ref("token")) == 5);
    values.reset(parse(body, "b"));
    CHECK(values.value(state_ref("token")) == 0);
}

// A spec that did not pass the validator may hold a range whose bounds read
// itself; the store then reads that bound as absent instead of recursing.
TEST_CASE("state: a range whose bound reads itself reads as unbounded") {
    const spec::Spec sp = parse(R"(
[[control]]
id = "win"
kind = "range"
bind = "value"
options = { min = { control = "win", field = "lo" }, max = 10, value = [2, 8] }
)");
    ValueStore values;
    values.reset(sp);
    const nlohmann::json v = values.control_value("win");
    REQUIRE(v.is_array());
    CHECK(v[1].get<double>() == 8.0);
    CHECK(values.set_control("win", nlohmann::json::array({1, 9}), CommandOrigin::ui, nullptr).ok());
}

// selection.set on a program's item: the item map's fields of the version at the cursor, the pick
// event and the select state, with each refusal of a program, a panel, an item and a version.
TEST_CASE("state: selection.set picks a program's item and refuses what the program does not draw") {
    namespace ex = caliper_test_pieces;
    spec::Spec sp = parse(R"(
[[state]]
id = "picked_head"
type = "integer"
min = 0
max = 63

[[panel]]
id = "world"
primitive = "scene"
bind = { scene = "live:model" }
options = { select = { state = "picked_head", field = "head" } }

[[panel]]
id = "side"
primitive = "scene"
bind = { scene = "live:model" }
)");
    spec::Piece piece = ex::ring_arcs();
    piece.hash = programs::piece_hash(piece);
    sp.pieces.push_back(piece);
    const auto program = [&](const char* id, const char* panel, const std::string& hash) {
        spec::Program p;
        p.id = id;
        spec::PieceRef pin;
        pin.name = piece.name;
        pin.version = piece.version;
        pin.hash = hash;
        pin.from = spec::PieceFrom::inline_piece;
        p.piece = pin;
        p.panel = panel;
        sp.programs.push_back(p);
    };
    program("tower_arcs", "world", piece.hash);
    program("side_arcs", "side", piece.hash);
    program("stale_arcs", "world", "sha256:" + std::string(64, '0'));
    program("short_arcs", "world", piece.hash);
    ValueStore values;
    values.reset(sp);
    StepHistory history;
    history.configure(sp);
    Recorder recorder;
    InstrumentContext ic{sp, true, values, history, &recorder};
    const auto run = [&](const char* name, const json& args) {
        CommandInvocation inv;
        inv.name = name;
        inv.args = args;
        const CommandDescriptor* d = find_command_descriptor(name);
        REQUIRE(d != nullptr);
        const CommandContext ctx{inv, *d};
        return selection_command(ctx, ic);
    };
    // Before the program published a version, nothing is picked.
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "tower_arcs"}, {"item", 0}}).error.code == "selection.bad_item");
    const ex::ArcResources res = ex::arc_resources();
    ResourceVersion v;
    v.step = 0;
    v.count = std::uint32_t(res.arcs.size());
    v.stride = 24;
    const auto* b = reinterpret_cast<const std::uint8_t*>(res.arcs.data());
    v.bytes = std::make_shared<const std::vector<std::uint8_t>>(b, b + res.arcs.size() * sizeof(ex::Arc));
    history.append_resource("tower_arcs", "arcs", 4, {}, v);

    const CommandOutput out = run("selection.set", {{"panel", "world"}, {"program", "tower_arcs"}, {"item", 2}});
    REQUIRE_MESSAGE(out.error.code.empty(), out.error.message);
    const ex::Arc& a = res.arcs[2];
    const json& f = out.result["fields"];
    CHECK(f["program"] == "tower_arcs");
    CHECK(f["item"] == 2);
    CHECK(f["step"] == 0);
    CHECK(f["layer"] == a.layer);
    CHECK(f["head"] == a.head);
    CHECK(f["fields"]["from_token"] == a.from_token);
    CHECK(f["fields"]["to_token"] == a.to_token);
    CHECK(f["fields"]["weight"] == doctest::Approx(a.weight));
    CHECK(f["index"] == json::array({a.layer, a.head, a.from_token, a.to_token}));
    CHECK(values.value(state_ref("picked_head")) == a.head);
    bool picked = false;
    for (const Event& e : recorder.events)
        picked = picked || (e.kind == EventKind::pick && e.fields["program"] == "tower_arcs" && e.fields["item"] == 2);
    CHECK(picked);
    CHECK(run("selection.get", {{"panel", "world"}}).result["fields"] == f);

    // The refusals: a program the spec lacks, one that draws in another panel, an item past the
    // version's count or past 32 bits, and a pin whose hash the piece does not have.
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "nope"}, {"item", 0}}).error.code == "program.not_found");
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "side_arcs"}, {"item", 0}}).error.code ==
          "selection.bad_item");
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "tower_arcs"}, {"item", res.arcs.size()}}).error.code ==
          "selection.bad_item");
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "tower_arcs"}, {"item", 4294967296ull}}).error.code ==
          "selection.bad_item");
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "tower_arcs"}, {"item", -1}}).error.code ==
          "selection.bad_item");
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "tower_arcs"}}).error.code == "selection.bad_item");
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "stale_arcs"}, {"item", 0}}).error.code ==
          "piece.hash_mismatch");
    // A version whose bytes hold fewer elements than its count: the item past the bytes is refused.
    ResourceVersion shortv = v;
    shortv.bytes = std::make_shared<const std::vector<std::uint8_t>>(b, b + 3 * sizeof(ex::Arc));
    history.append_resource("short_arcs", "arcs", 4, {}, shortv);
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "short_arcs"}, {"item", 2}}).error.code.empty());
    CHECK(run("selection.set", {{"panel", "world"}, {"program", "short_arcs"}, {"item", 3}}).error.code ==
          "selection.bad_item");
}
