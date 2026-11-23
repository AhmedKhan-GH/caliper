// The app's program host over the builder session (AppProgramHost in
// src/host/live/live_programs.h, design/gpu-programs.md, Program Status and
// Program Events): the status items that program.status returns carry the
// script's cadence, and a program that the script stopped for its budget
// shows program.over_budget in its status item and in a `program` event.
// Compiled into caliper_live_tests only with CALIPER_WITH_BUILDER.
#include <doctest/doctest.h>

#include "live/live_programs.h"

#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/program_host.h>

#include "gfx/example_pieces.h"

#include <string>
#include <vector>

namespace {

namespace cb = caliper_builder;

struct Link final : caliper_live::ProgramLink {
    uint32_t next = 1;
    std::vector<caliper_net::wire::Program> sent;
    std::vector<caliper_live::ProgramAnswer> answers;
    bool ready() override { return true; }
    uint64_t epoch() override { return 1; }
    uint32_t send(caliper_net::wire::Program m) override {
        m.request_id = next++;
        sent.push_back(std::move(m));
        return sent.back().request_id;
    }
    std::vector<caliper_live::ProgramAnswer> take() override {
        std::vector<caliper_live::ProgramAnswer> out;
        out.swap(answers);
        return out;
    }
};

cb::spec::Program pixels(const std::string& id) {
    cb::spec::Program p;
    p.id = id;
    p.wgsl = "@compute @workgroup_size(64) fn run() {} // " + id;
    p.inputs = {{"w", "live:0.weight"}};
    p.output.kind = cb::spec::OutputKind::pixels;
    p.output.width = 16;
    p.output.height = 16;
    return p;
}

// The script's status item of a running program with its cadence, or of a
// program stopped with `code`.
caliper_net::Json script_item(const caliper_live::WantedProgram& w, const std::string& state,
                              const std::string& code = "") {
    caliper_net::Json o = caliper_net::Json::object();
    o.set("id", w.id);
    o.set("kind", "pixels");
    o.set("state", state);
    o.set("every", 48);
    o.set("every_min", 1);
    o.set("last_pass_ms", 0.25);
    o.set("last_cost_ms", 0.75);
    o.set("step_ms", 0.5);
    o.set("overhead_share", 0.03125);
    o.set("passes_per_s", 40.0);
    o.set("last_step", 960);
    o.set("runs", 20);
    o.set("sha256", w.sha256);
    if (!code.empty()) {
        caliper_net::Json e = caliper_net::Json::object();
        e.set("code", code);
        e.set("message", "A pass adds 9 ms to a step of 0.5 ms, more than the ceiling allows.");
        e.set("line", nullptr);
        e.set("column", nullptr);
        o.set("error", e);
    }
    return o;
}

caliper_live::ProgramAnswer answer(std::vector<caliper_net::Json> items) {
    caliper_live::ProgramAnswer a;
    a.programs = caliper_net::Json::array();
    for (auto& x : items) a.programs.push(std::move(x));
    return a;
}

}  // namespace

TEST_CASE("live/programs: program.status carries the script's cadence, and over_budget stops with a status and an event") {
    cb::spec::Spec spec;
    spec.programs = {pixels("heat"), pixels("image")};
    const auto heat = caliper_live::AppProgramHost::wanted_program(spec, spec.programs[0]);
    const auto image = caliper_live::AppProgramHost::wanted_program(spec, spec.programs[1]);
    caliper_live::AppProgramHost host;
    cb::EventRing ring;
    Link link;
    host.reconcile(spec);
    host.pump(link, &ring);
    REQUIRE(link.sent.size() == 2);

    link.answers.push_back(answer({script_item(heat, "running"), script_item(image, "running")}));
    host.pump(link, &ring);
    auto st = host.status();
    REQUIRE(st.size() == 2);
    CHECK(st[0].host == "script");
    CHECK(st[0].state == "running");
    CHECK(st[0].every.value_or(0) == 48);
    CHECK(st[0].last_cost_ms.value_or(0) == 0.75);
    CHECK(st[0].step_ms.value_or(0) == 0.5);
    CHECK(st[0].overhead_share.value_or(0) == 0.03125);
    CHECK(st[0].passes_per_s.value_or(0) == 40.0);
    const nlohmann::json j = cb::program_status_json(st[0]);
    CHECK(j["overhead_share"] == 0.03125);
    CHECK(j["passes_per_s"] == 40.0);
    CHECK(j["error"].is_null());
    const std::uint64_t cursor = ring.latest();   // past the two `compiled` events

    // Even the ceiling of N costs more than the budget: the script stops every
    // program with program.over_budget, and training goes on.
    link.answers.push_back(answer({script_item(heat, "stopped", cb::programs::kProgramOverBudget),
                                   script_item(image, "stopped", cb::programs::kProgramOverBudget)}));
    host.pump(link, &ring);
    st = host.status();
    REQUIRE(st.size() == 2);
    for (const cb::ProgramStatus& s : st) {
        CAPTURE(s.id);
        CHECK(s.state == "stopped");
        REQUIRE(s.error.has_value());
        CHECK(s.error->code == cb::programs::kProgramOverBudget);
        CHECK(s.error->line == 0);
        CHECK(s.overhead_share.value_or(0) == 0.03125);   // the item still shows why
        CHECK(cb::program_status_json(s)["error"]["code"] == "program.over_budget");
    }
    const cb::EventRead read = ring.read(cursor, {{cb::EventKind::program}, {}}, 8);
    REQUIRE(read.events.size() == 2);
    std::vector<std::string> ids;
    for (const cb::Event& e : read.events) {
        const nlohmann::json ev = cb::event_json(e);
        CHECK(ev["kind"] == "program");
        CHECK(ev["change"] == "stopped");
        CHECK(ev["host"] == "script");
        CHECK(ev["code"] == "program.over_budget");
        ids.push_back(ev["program"].get<std::string>());
    }
    CHECK(ids == std::vector<std::string>{"heat", "image"});

    host.pump(link, &ring);   // nothing new: no second event
    CHECK(ring.read(cursor, {{cb::EventKind::program}, {}}, 8).events.size() == 2);
}

// A changed value of a reference in a program's params is a view change
// (design/instrument-interface.md, Decisions): it goes to the script as
// PROGRAM op params with the values alone, never as a set that installs,
// replaces or declares the program again, and a value outside the declared
// range of its field in struct Params is refused and not sent.
TEST_CASE("live/programs: a params change sends only values, and a value outside the declared range is refused") {
    cb::spec::Spec spec;
    cb::spec::Program heat = pixels("heat");
    heat.wgsl = "struct Params { level: u32, gain: f32 }\n"
                "@group(0) @binding(9) var<uniform> params: Params;\n"
                "@compute @workgroup_size(64) fn run() {}\n";
    cb::spec::Reference level, gain;
    level.kind = gain.kind = cb::spec::Reference::Kind::state;
    level.target = "level";
    gain.target = "gain";
    heat.params = {{"level", level}, {"gain", gain}};
    spec.programs = {heat};
    const auto wanted = caliper_live::AppProgramHost::wanted_program(spec, spec.programs[0]);
    caliper_live::AppProgramHost host;
    cb::EventRing ring;
    Link link;
    host.reconcile(spec);
    host.pump(link, &ring);
    REQUIRE(link.sent.size() == 1);
    CHECK(link.sent[0].op == CALIPER_NET_PROGRAM_SET);
    link.answers.push_back(answer({script_item(wanted, "running")}));
    host.pump(link, &ring);
    const std::size_t installed = link.sent.size();
    auto sets = [&] {
        std::size_t n = 0;
        for (const auto& m : link.sent) n += m.op == CALIPER_NET_PROGRAM_SET || m.op == CALIPER_NET_PROGRAM_REMOVE;
        return n;
    };

    double lv = 3, gv = 0.5;
    const caliper_live::AppProgramHost::Evaluate evaluate = [&](const cb::spec::Reference& r) -> nlohmann::json {
        return r.target == "level" ? lv : gv;
    };
    CHECK(host.refresh(evaluate, nullptr));
    host.pump(link, &ring);
    REQUIRE(link.sent.size() == installed + 1);
    const caliper_net::wire::Program& sent = link.sent.back();
    CHECK(sent.op == CALIPER_NET_PROGRAM_PARAMS);
    CHECK(sent.id == "heat");
    CHECK(sent.declaration.is_null());
    CHECK_FALSE(sent.wgsl.has_value());
    CHECK_FALSE(sent.sha256.has_value());
    CHECK_FALSE(sent.scene.has_value());
    REQUIRE(sent.params.find("level"));
    CHECK(sent.params.find("level")->as_double() == 3);
    CHECK(sent.params.find("gain")->as_double() == 0.5);
    CHECK(sets() == 1);
    REQUIRE(host.status().size() == 1);
    CHECK(host.status()[0].sha256 == wanted.sha256);

    // u32 takes no negative value and none past 2^32 - 1, and f32 none past its range.
    for (const auto& [l, g] : {std::pair<double, double>{-1, 0.5}, {4294967296.0, 0.5}, {3, 1e39}}) {
        lv = l;
        gv = g;
        CHECK_FALSE(host.refresh(evaluate, nullptr));
    }
    host.pump(link, &ring);
    CHECK(link.sent.size() == installed + 1);
    CHECK(host.refused_params() == 3);

    // A value inside the range goes out, still as params alone.
    gv = 2;
    CHECK(host.refresh(evaluate, nullptr));
    host.pump(link, &ring);
    REQUIRE(link.sent.size() == installed + 2);
    CHECK(link.sent.back().op == CALIPER_NET_PROGRAM_PARAMS);
    CHECK(link.sent.back().params.find("level")->as_double() == 3);
    CHECK(link.sent.back().params.find("gain")->as_double() == 2);
    CHECK(sets() == 1);
}

// A program of a foreign spec (design/gpu-programs.md, Provenance;
// design/programmable-graphics.md, Piece Sources) never goes out to the
// script, which would compile it and relay its draw half to the viewers,
// before provenance.reviewed holds its hash: the source's SHA-256 for the
// one-pass form, the piece's hash for an instance. Until then its state is
// review, without an error and without an event.
TEST_CASE("live/programs: a foreign spec's programs and pieces wait for a review of their hash before they go out") {
    struct PieceLink final : caliper_live::ProgramLink {
        Link inner;
        bool ready() override { return true; }
        uint64_t epoch() override { return 1; }
        uint32_t send(caliper_net::wire::Program m) override { return inner.send(std::move(m)); }
        std::vector<caliper_live::ProgramAnswer> take() override { return inner.take(); }
        bool takes_pieces() override { return true; }
    };
    cb::spec::Spec spec;
    cb::spec::Piece field = caliper_test_pieces::attention_field();
    const std::string field_hash = cb::programs::piece_hash(field);
    field.hash = field_hash;
    spec.pieces = {field};
    cb::spec::Program instance;
    instance.id = "field";
    cb::spec::PieceRef pin;
    pin.name = field.name;
    pin.version = field.version;
    pin.hash = field_hash;
    pin.from = cb::spec::PieceFrom::inline_piece;
    instance.piece = pin;
    instance.panel = std::string("pattern");
    instance.bind = {{"attn0", "live:0.attn"}, {"attn1", "live:1.attn"}};
    spec.programs = {pixels("heat"), instance};
    spec.provenance.foreign = true;
    spec.provenance.from = std::string("a shared instrument");

    caliper_live::AppProgramHost host;
    cb::EventRing ring;
    PieceLink link;
    host.reconcile(spec);
    host.pump(link, &ring);
    CHECK(link.inner.sent.empty());
    auto st = host.status();
    REQUIRE(st.size() == 2);
    for (const cb::ProgramStatus& s : st) {
        CAPTURE(s.id);
        CHECK(s.state == "review");
        CHECK(s.host == "none");
        CHECK(!s.error.has_value());
    }
    CHECK(ring.read(0, {{cb::EventKind::program}, {}}, 8).events.empty());

    // A review of another hash for the target does not count.
    spec.provenance.reviewed = {{"program:heat", std::string(64, '0')},
                                {"program:field", cb::programs::sha256_hex(spec.programs[0].wgsl)}};
    host.reconcile(spec);
    host.pump(link, &ring);
    CHECK(link.inner.sent.empty());

    // The reviewed hashes let both go out: the source's for heat, the piece's for the instance.
    spec.provenance.reviewed = {{"program:heat", cb::programs::sha256_hex(spec.programs[0].wgsl)},
                                {"program:field", field_hash.substr(field_hash.find(':') + 1)}};
    host.reconcile(spec);
    host.pump(link, &ring);
    REQUIRE(link.inner.sent.size() == 2);
    CHECK(link.inner.sent[0].id == "heat");
    CHECK(link.inner.sent[1].id == "field");
    for (const cb::ProgramStatus& s : host.status()) CHECK(s.state == "waiting");

    // A changed source of a reviewed program waits again, and the script removes the old one.
    spec.programs[0].wgsl += "\n// changed";
    host.reconcile(spec);
    host.pump(link, &ring);
    CHECK(host.status()[0].state == "review");
    CHECK(link.inner.sent.back().op == CALIPER_NET_PROGRAM_REMOVE);
    CHECK(link.inner.sent.back().id == "heat");
}

// A pin needs its hash (design/programmable-graphics.md, Versions and Pins).
// The editor writes it only into the pins of program.create and
// program.patch, thus a pin without one in a spec file is refused with
// spec.missing_key and never goes out, and the pin with the piece's hash goes
// out.
TEST_CASE("live/programs: a pin without a hash is a missing key, not a hash mismatch") {
    struct PieceLink final : caliper_live::ProgramLink {
        Link inner;
        bool ready() override { return true; }
        uint64_t epoch() override { return 1; }
        uint32_t send(caliper_net::wire::Program m) override { return inner.send(std::move(m)); }
        std::vector<caliper_live::ProgramAnswer> take() override { return inner.take(); }
        bool takes_pieces() override { return true; }
    };
    cb::spec::Spec spec;
    cb::spec::Piece field = caliper_test_pieces::attention_field();
    field.hash = cb::programs::piece_hash(field);
    spec.pieces = {field};
    cb::spec::Program instance;
    instance.id = "field";
    cb::spec::PieceRef pin;
    pin.name = field.name;
    pin.version = field.version;
    pin.from = cb::spec::PieceFrom::inline_piece;
    instance.piece = pin;
    instance.panel = std::string("pattern");
    instance.bind = {{"attn0", "live:0.attn"}, {"attn1", "live:1.attn"}};
    spec.programs = {instance};

    caliper_live::AppProgramHost host;
    cb::EventRing ring;
    PieceLink link;
    host.reconcile(spec);
    host.pump(link, &ring);
    CHECK(link.inner.sent.empty());
    auto st = host.status();
    REQUIRE(st.size() == 1);
    CHECK(st[0].state == "refused");
    REQUIRE(st[0].error.has_value());
    CHECK(st[0].error->code == "spec.missing_key");
    CHECK(st[0].error->message.find("no hash") != std::string::npos);

    spec.programs[0].piece->hash = field.hash;
    host.reconcile(spec);
    host.pump(link, &ring);
    REQUIRE(link.inner.sent.size() == 1);
    CHECK(link.inner.sent[0].id == "field");
    CHECK(host.status()[0].state == "waiting");
}
