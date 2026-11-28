// The app's program host over the builder session (AppProgramHost in
// src/host/live/live_programs.h, design/gpu-programs.md, Program Status and
// Program Events): the status items that program.status returns carry the
// script's cadence, and a program over its training budget shows
// program.over_budget as a recoverable warning in its status item and in a
// `program` event while it keeps running.
// Compiled into caliper_live_tests only with CALIPER_WITH_BUILDER.
#include <doctest/doctest.h>

#include "live/live_programs.h"

#include <caliper_builder/programs/sandbox.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/program_host.h>

#include "gfx/example_pieces.h"

#include <memory>
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

// The script's status item of a running program with its cadence, or with an
// error or warning code.
caliper_net::Json script_item(const caliper_live::WantedProgram& w, const std::string& state,
                              const std::string& error_code = "", const std::string& warning_code = "") {
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
    if (!error_code.empty()) {
        caliper_net::Json e = caliper_net::Json::object();
        e.set("code", error_code);
        e.set("message", "The pass timed out on the GPU.");
        e.set("line", nullptr);
        e.set("column", nullptr);
        o.set("error", e);
    }
    if (!warning_code.empty()) {
        caliper_net::Json warn = caliper_net::Json::object();
        warn.set("code", warning_code);
        warn.set("message", "A pass adds 9 ms to a step of 0.5 ms, more than the ceiling allows.");
        warn.set("line", nullptr);
        warn.set("column", nullptr);
        o.set("warning", warn);
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

TEST_CASE("live/programs: program.status carries the script's cadence, and over_budget warns without stopping") {
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
    CHECK(j["warning"].is_null());
    const std::uint64_t cursor = ring.latest();   // past the two `compiled` events

    // Even the ceiling of N costs more than the budget: the script warns with
    // program.over_budget while the programs keep running.
    link.answers.push_back(answer({script_item(heat, "running", "", cb::programs::kProgramOverBudget),
                                   script_item(image, "running", "", cb::programs::kProgramOverBudget)}));
    host.pump(link, &ring);
    st = host.status();
    REQUIRE(st.size() == 2);
    for (const cb::ProgramStatus& s : st) {
        CAPTURE(s.id);
        CHECK(s.state == "running");
        CHECK(!s.error.has_value());
        REQUIRE(s.warning.has_value());
        CHECK(s.warning->code == cb::programs::kProgramOverBudget);
        CHECK(s.warning->line == 0);
        CHECK(s.overhead_share.value_or(0) == 0.03125);   // the item still shows why
        CHECK(cb::program_status_json(s)["warning"]["code"] == "program.over_budget");
    }
    const cb::EventRead read = ring.read(cursor, {{cb::EventKind::program}, {}}, 8);
    REQUIRE(read.events.size() == 2);
    std::vector<std::string> ids;
    for (const cb::Event& e : read.events) {
        const nlohmann::json ev = cb::event_json(e);
        CHECK(ev["kind"] == "program");
        CHECK(ev["change"] == "warned");
        CHECK(ev["host"] == "script");
        CHECK(ev["code"] == "program.over_budget");
        ids.push_back(ev["program"].get<std::string>());
    }
    CHECK(ids == std::vector<std::string>{"heat", "image"});

    host.pump(link, &ring);   // nothing new: no second event
    CHECK(ring.read(cursor, {{cb::EventKind::program}, {}}, 8).events.size() == 2);

    // Recovery clears the warning without pushing another event.
    link.answers.push_back(answer({script_item(heat, "running"), script_item(image, "running")}));
    host.pump(link, &ring);
    for (const cb::ProgramStatus& s : host.status()) {
        CHECK(s.state == "running");
        CHECK(!s.warning.has_value());
    }
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

TEST_CASE("live/programs: only a resource that the pinned piece publishes, in its declared form, reaches the history") {
    struct ResourceLink final : caliper_live::ProgramLink {
        std::vector<caliper_live::ReceivedResource> incoming;
        bool ready() override { return true; }
        uint64_t epoch() override { return 1; }
        uint32_t send(caliper_net::wire::Program) override { return 1; }
        std::vector<caliper_live::ProgramAnswer> take() override { return {}; }
        bool takes_pieces() override { return true; }
        std::vector<caliper_live::ReceivedResource> take_resources() override {
            std::vector<caliper_live::ReceivedResource> out;
            out.swap(incoming);
            return out;
        }
    };
    cb::spec::Spec spec;
    cb::spec::Piece arcs = caliper_test_pieces::ring_arcs();   // heads: f32 x 8; arcs: 24 bytes x 4096
    arcs.hash = cb::programs::piece_hash(arcs);
    spec.pieces = {arcs};
    cb::spec::Program instance;
    instance.id = "tower";
    cb::spec::PieceRef pin;
    pin.name = arcs.name;
    pin.version = arcs.version;
    pin.hash = arcs.hash;
    pin.from = cb::spec::PieceFrom::inline_piece;
    instance.piece = pin;
    instance.bind = {{"attn0", "live:0.attn"}, {"attn1", "live:1.attn"}};
    spec.programs = {instance};
    caliper_live::AppProgramHost host;
    host.reconcile(spec);
    cb::StepHistory history;
    history.configure(spec);
    auto resource = [](const std::string& program, const std::string& name, uint32_t count, uint32_t stride,
                       int64_t step) {
        caliper_live::ReceivedResource r;
        r.program = program;
        r.resource = name;
        r.step = step;
        r.count = count;
        r.stride = stride;
        r.bytes = std::make_shared<const std::vector<uint8_t>>(std::size_t(count) * stride, uint8_t(7));
        return r;
    };
    ResourceLink link;
    link.incoming = {resource("tower", "heads", 8, 4, 10),       // as declared
                     resource("tower", "arcs", 100, 24, 10),     // fewer than its count
                     resource("tower", "spare", 8, 4, 10),       // a name that the piece does not publish
                     resource("tower", "heads", 9, 4, 11),       // more than its count
                     resource("tower", "arcs", 100, 32, 11),     // another stride
                     resource("other", "heads", 8, 4, 10)};      // a program that the spec does not have
    caliper_live::ReceivedResource texture = resource("tower", "heads", 4, 4, 12);
    texture.width = texture.height = 2;                          // a texture where a buffer is declared
    link.incoming.push_back(texture);
    caliper_live::ReceivedResource short_bytes = resource("tower", "heads", 8, 4, 13);
    short_bytes.bytes = std::make_shared<const std::vector<uint8_t>>(std::size_t(16), uint8_t(0));
    link.incoming.push_back(short_bytes);
    host.deliver_resources(link, history);
    CHECK(host.resources_received() == 8);
    CHECK(host.resources_refused() == 6);
    std::vector<std::string> keys;
    for (const cb::KeptResource& k : history.kept_resources()) keys.push_back(k.key);
    CHECK(keys == std::vector<std::string>{"tower/arcs", "tower/heads"});
    CHECK(history.resource_at("tower/heads", 100)->step == 10);
    CHECK(history.resource_at("tower/arcs", 100)->count == 100);
}

TEST_CASE("live/programs: a texture's version reaches the history with its format and a size within its largest, a "
          "narrow buffer's with its elements") {
    struct ResourceLink final : caliper_live::ProgramLink {
        std::vector<caliper_live::ReceivedResource> incoming;
        bool ready() override { return true; }
        uint64_t epoch() override { return 1; }
        uint32_t send(caliper_net::wire::Program) override { return 1; }
        std::vector<caliper_live::ProgramAnswer> take() override { return {}; }
        bool takes_pieces() override { return true; }
        std::vector<caliper_live::ReceivedResource> take_resources() override {
            std::vector<caliper_live::ReceivedResource> out;
            out.swap(incoming);
            return out;
        }
    };
    cb::spec::Spec spec;
    cb::spec::Piece arcs = caliper_test_pieces::ring_arcs();
    cb::spec::Resource grey;   // r8unorm, at most 8 by 4 and 20 texels
    grey.id = "grey";
    grey.kind = cb::spec::ResourceKind::texture;
    grey.format = "r8unorm";
    grey.width = 8;
    grey.height = 4;
    grey.texels = 20;
    cb::spec::Resource ids;    // 9 u16 elements
    ids.id = "ids";
    ids.layout.element = "u16";
    ids.count = 9;
    arcs.resources.push_back(grey);
    arcs.resources.push_back(ids);
    arcs.hash = cb::programs::piece_hash(arcs);
    spec.pieces = {arcs};
    cb::spec::Program instance;
    instance.id = "tower";
    cb::spec::PieceRef pin;
    pin.name = arcs.name;
    pin.version = arcs.version;
    pin.hash = arcs.hash;
    pin.from = cb::spec::PieceFrom::inline_piece;
    instance.piece = pin;
    instance.bind = {{"attn0", "live:0.attn"}, {"attn1", "live:1.attn"}};
    spec.programs = {instance};
    caliper_live::AppProgramHost host;
    host.reconcile(spec);
    cb::StepHistory history;
    history.configure(spec);
    auto texture = [](uint32_t w, uint32_t h, uint32_t words, const char* format, int64_t step) {
        caliper_live::ReceivedResource r;
        r.program = "tower";
        r.resource = "grey";
        r.step = step;
        r.count = words;
        r.stride = 4;
        r.width = w;
        r.height = h;
        r.format = format;
        r.bytes = std::make_shared<const std::vector<uint8_t>>(std::size_t(words) * 4, uint8_t(7));
        return r;
    };
    auto narrow = [](std::optional<uint32_t> elements, uint32_t words, int64_t step) {
        caliper_live::ReceivedResource r;
        r.program = "tower";
        r.resource = "ids";
        r.step = step;
        r.count = words;
        r.stride = 4;
        r.elements = elements;
        r.bytes = std::make_shared<const std::vector<uint8_t>>(std::size_t(words) * 4, uint8_t(1));
        return r;
    };
    ResourceLink link;
    link.incoming = {texture(5, 3, 4, "r8unorm", 10),     // 15 texels in 4 words: kept
                     texture(0, 0, 0, "", 11),            // a version without texels: kept
                     texture(5, 3, 15, "", 12),           // rgba8unorm where r8unorm is declared
                     texture(9, 1, 3, "r8unorm", 13),     // wider than the largest side
                     texture(7, 3, 6, "r8unorm", 14),     // 21 texels, above 20
                     texture(5, 3, 5, "r8unorm", 15),     // words that do not match the size
                     narrow(5u, 3, 10),                   // 5 elements in 3 words: kept
                     narrow(10u, 5, 11),                  // above 9 elements
                     narrow(4u, 3, 12),                   // 4 elements fill 2 words
                     narrow(std::nullopt, 3, 13)};        // no element count
    host.deliver_resources(link, history);
    CHECK(host.resources_received() == 10);
    CHECK(host.resources_refused() == 7);
    const auto kept = history.resource_at("tower/grey", 10);
    REQUIRE(kept);
    CHECK(kept->width == 5);
    CHECK(kept->height == 3);
    CHECK(kept->format == "r8unorm");
    CHECK(kept->count == 4);
    CHECK(history.resource_at("tower/grey", 11)->count == 0);
    const auto n = history.resource_at("tower/ids", 100);
    REQUIRE(n);
    CHECK(n->elements == std::optional<uint32_t>(5));
    CHECK(n->count == 3);
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
    REQUIRE(link.inner.sent[1].piece.find("from") != nullptr);
    CHECK(link.inner.sent[1].piece.find("from")->as_string() == "inline");
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
