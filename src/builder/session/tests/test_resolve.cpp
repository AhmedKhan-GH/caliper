// Tests of the resolved data of a panel: tensors after their transforms down a
// chain of `from`, columns and tables of catalog tables and `sql` sources, the
// table panel's columns option, the bounds on rows, cells and tensor sizes,
// and the sources the session cannot read yet.
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/panel_data.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/session/session.h>
#include <caliper_builder/session/values.h>

#include <doctest/doctest.h>

#include "helpers.h"

#include <cstring>
#include <limits>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

std::vector<double> values_of(const TensorData& t) {
    std::vector<double> out;
    for (uint64_t i = 0; i < t.numel(); ++i) out.push_back(t.element(i));
    return out;
}

struct Fixture {
    std::string ws;
    std::unique_ptr<Session> s;
    explicit Fixture(const std::string& name, ResolveLimits limits = {}) {
        ws = fresh_dir(name) + "/workspace";
        std::string csv = "step,loss,label\n";
        for (int i = 0; i < 10; ++i) csv += std::to_string(i) + "," + std::to_string(10 - i) + ".5,row" + std::to_string(i) + "\n";
        write_text(ws + "/data/metrics.csv", csv);
        // [2, 2, 3]: two layers of 2 x 3.
        write_safetensors_f32(ws + "/data/acts.safetensors", "x", {2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        SessionOptions options;
        options.registry.check_results = true;
        options.resolve = limits;
        auto opened = Session::open(ws, options);
        REQUIRE(opened.error.ok());
        s = std::move(opened.session);
        ok(*s, "data.import", {{"paths", {"data/metrics.csv", "data/acts.safetensors"}}});
        ok(*s, "spec.new", {{"id", "r"}});
    }
};

}  // namespace

TEST_CASE("a tensor binding resolves to the tensor after every transform down the chain") {
    Fixture f("tensor");
    ok(*f.s, "source.add", {{"id", "acts"}, {"catalog", "tensor:acts_x"},
                            {"transform", {{{"op", "slice"}, {"axis", 0}, {"index", 1}}}}});  // [[7,8,9],[10,11,12]]
    ok(*f.s, "source.add", {{"id", "mean"}, {"from", "acts"},
                            {"transform", {{{"op", "reduce"}, {"axis", 1}, {"fn", "mean"}}}}});  // [8, 11]
    ok(*f.s, "source.add", {{"id", "unit"}, {"from", "mean"}, {"transform", {{{"op", "normalise"}, {"mode", "minmax"}}}}});
    ok(*f.s, "panel.create", {{"id", "heat"}, {"primitive", "heatmap"}, {"bind", {{"values", "acts"}}}});
    ok(*f.s, "panel.create", {{"id", "line"}, {"primitive", "series"}, {"bind", {{"x", "mean"}, {"y", {"unit", "mean"}}}}});

    const PanelData heat = f.s->resolve("heat");
    REQUIRE_MESSAGE(heat.error.ok(), heat.error.message);
    REQUIRE(heat.slots.size() == 1);
    CHECK(heat.slots[0].kind == SlotData::Kind::tensor);
    CHECK(heat.slots[0].tensor.shape() == std::vector<int64_t>{2, 3});
    CHECK(values_of(heat.slots[0].tensor) == std::vector<double>{7, 8, 9, 10, 11, 12});

    const PanelData line = f.s->resolve("line");
    REQUIRE_MESSAGE(line.error.ok(), line.error.message);
    REQUIRE(line.slots.size() == 3);
    CHECK(line.slots[0].slot == "x");
    CHECK(values_of(line.slots[0].tensor) == std::vector<double>{8, 11});
    CHECK(line.slots[1].slot == "y");
    CHECK(line.slots[1].index == 0);
    CHECK(values_of(line.slots[1].tensor) == std::vector<double>{0, 1});
    CHECK(line.slots[2].index == 1);
    CHECK(line.slots[2].binding == "mean");
    const json summary = panel_data_summary(line);
    CHECK(summary["slots"][1]["values"] == json{0.0, 1.0});
}

TEST_CASE("columns and tables resolve through the store's bounded query") {
    Fixture f("table");
    ok(*f.s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    ok(*f.s, "source.add", {{"id", "q"}, {"sql", "SELECT step * 2 AS twice, label FROM metrics WHERE step < 3 ORDER BY step"}});
    ok(*f.s, "source.add", {{"id", "alias"}, {"from", "m"}});
    ok(*f.s, "panel.create", {{"id", "line"}, {"primitive", "series"}, {"bind", {{"x", "m.step"}, {"y", {"m.loss"}}}}});
    ok(*f.s, "panel.create", {{"id", "alias"}, {"primitive", "histogram"}, {"bind", {{"values", "alias.loss"}}}});
    ok(*f.s, "panel.create", {{"id", "rows"}, {"primitive", "table"}, {"bind", {{"rows", "m"}}},
                              {"options", {{"columns", {"label", "step"}}}}});
    ok(*f.s, "panel.create", {{"id", "sql"}, {"primitive", "table"}, {"bind", {{"rows", "q"}}}});
    ok(*f.s, "panel.create", {{"id", "last"}, {"primitive", "text"}, {"bind", {{"value", "q.label"}}}});

    const PanelData line = f.s->resolve("line");
    REQUIRE_MESSAGE(line.error.ok(), line.error.message);
    REQUIRE(line.slots.size() == 2);
    CHECK(line.slots[0].kind == SlotData::Kind::column);
    CHECK(line.slots[0].columns[0].type == "BIGINT");
    CHECK(line.slots[0].values == json{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    CHECK(line.slots[1].values[0] == 10.5);
    CHECK(line.slots[1].values[9] == 1.5);
    // A source derived with `from` reads its root's rows.
    const PanelData alias = f.s->resolve("alias");
    REQUIRE_MESSAGE(alias.error.ok(), alias.error.message);
    CHECK(alias.slots[0].values == line.slots[1].values);

    const PanelData rows = f.s->resolve("rows");
    REQUIRE(rows.error.ok());
    CHECK(rows.slots[0].kind == SlotData::Kind::table);
    REQUIRE(rows.slots[0].columns.size() == 2);
    CHECK(rows.slots[0].columns[0].name == "label");
    CHECK(rows.slots[0].values[2] == json{"row2", 2});

    const PanelData sql = f.s->resolve("sql");
    REQUIRE(sql.error.ok());
    CHECK(sql.slots[0].values == json{{0, "row0"}, {2, "row1"}, {4, "row2"}});
    const PanelData last = f.s->resolve("last");
    REQUIRE(last.error.ok());
    CHECK(last.slots[0].values == json{"row0", "row1", "row2"});
    CHECK(f.s->resolve("none").error.code == "panel.not_found");
}

TEST_CASE("the bounds cut rows and refuse a tensor that grows too large") {
    ResolveLimits limits;
    limits.max_rows = 4;
    limits.max_cells = 6;  // with two columns, three rows
    limits.max_tensor_bytes = 16;
    Fixture f("bounds", limits);
    ok(*f.s, "source.add", {{"id", "m"}, {"catalog", "table:metrics"}});
    ok(*f.s, "source.add", {{"id", "q"}, {"sql", "SELECT step, loss FROM metrics"}});
    ok(*f.s, "source.add", {{"id", "a"}, {"catalog", "tensor:acts_x"}, {"transform", {{{"op", "normalise"}, {"mode", "l2"}}}}});
    ok(*f.s, "panel.create", {{"id", "one"}, {"primitive", "series"}, {"bind", {{"y", "m.loss"}}}});
    ok(*f.s, "panel.create", {{"id", "two"}, {"primitive", "table"}, {"bind", {{"rows", "q"}}}});
    ok(*f.s, "panel.create", {{"id", "big"}, {"primitive", "histogram"}, {"bind", {{"values", "a"}}}});
    const PanelData one = f.s->resolve("one");
    REQUIRE(one.error.ok());
    CHECK(one.slots[0].values.size() == 4);
    CHECK(one.slots[0].truncated);
    const PanelData two = f.s->resolve("two");
    REQUIRE(two.error.ok());
    CHECK(two.slots[0].values.size() == 3);
    CHECK(two.slots[0].truncated);
    // A normalised f32 tensor of 12 elements is 48 bytes, past the 16 allowed.
    CHECK(f.s->resolve("big").error.code == "resolve.too_large");
}

TEST_CASE("sources the session cannot read answer clearly") {
    Fixture f("unavailable");
    ok(*f.s, "source.add", {{"id", "live"}, {"catalog", "live:attn.weights"}});
    ok(*f.s, "source.add", {{"id", "gone"}, {"catalog", "tensor:not_loaded"}});
    ok(*f.s, "source.add", {{"id", "bad"}, {"sql", "SELECT * FROM nowhere"}});
    ok(*f.s, "panel.create", {{"id", "a"}, {"primitive", "heatmap"}, {"bind", {{"values", "live"}}}});
    ok(*f.s, "panel.create", {{"id", "b"}, {"primitive", "heatmap"}, {"bind", {{"values", "gone"}}}});
    ok(*f.s, "panel.create", {{"id", "c"}, {"primitive", "table"}, {"bind", {{"rows", "bad"}}}});
    CHECK(f.s->resolve("a").error.code == "resolve.unavailable");
    CHECK(f.s->resolve("b").error.code == "catalog.not_found");
    CHECK(f.s->resolve("c").error.code == "resolve.query_failed");

    // Without an open spec there is nothing to resolve.
    ok(*f.s, "history.undo", {{"steps", 7}});
    CHECK(!f.s->open_spec());
    CHECK(f.s->resolve("a").error.code == "spec.no_open_spec");
}

TEST_CASE("resolve_panel follows a spec that the validator would refuse without looping") {
    Fixture f("cycle");
    spec::Spec s;
    s.instrument.id = "c";
    spec::Source a;
    a.id = "a";
    a.from = "b";
    spec::Source b;
    b.id = "b";
    b.from = "a";
    s.sources = {a, b};
    spec::Panel p;
    p.id = "p";
    p.primitive = spec::Primitive::heatmap;
    p.bind["values"] = spec::SlotValue{{"a"}, false};
    s.panels = {p};
    CHECK(resolve_panel(s, "p", f.s->store()).error.code == "resolve.cycle");
    s.sources[1].from = "zz";
    CHECK(resolve_panel(s, "p", f.s->store()).error.code == "resolve.unknown_source");
}

// ---- the time cursor (design/instrument-interface.md, Latest at the Cursor
// and Window Step) -------------------------------------------------------------

namespace {

// A live view that the app serves: its newest value, a new generation on
// each put.
class LiveFake : public CatalogDataSource {
public:
    std::uint64_t generation(const std::string& id) override { return id == "live:act" ? gen : 0; }
    ServedData read(const std::string& id, const ServedRequest&) override {
        ServedData out;
        if (id != "live:act" || !newest.valid()) {
            out.error = Error("resolve.unavailable", "No data yet.");
            return out;
        }
        out.generation = gen;
        out.tensor = newest;
        return out;
    }
    void put(const TensorData& t) {
        newest = t;
        ++gen;
    }
    TensorData newest;
    std::uint64_t gen = 0;
};

TensorData row(float v, std::int64_t n = 2) {
    TensorResult r = TensorData::allocate(DType::f32, {n});
    REQUIRE(r.error.ok());
    for (std::int64_t i = 0; i < n; ++i) reinterpret_cast<float*>(r.tensor.mutable_data())[i] = v + float(i);
    return r.tensor;
}

const char* kCursorSpec = R"(caliper_spec = 2
[instrument]
id = "cursor"

[[source]]
id = "act"
catalog = "live:act"

[[source]]
id = "first"
from = "act"
transform = [{ op = "slice", axis = 0, index = 0 }]

[[source]]
id = "recent"
from = "act"
transform = [{ op = "window", steps = 2 }]

[[panel]]
id = "now"
primitive = "series"
bind = { y = "act" }

[[panel]]
id = "derived"
primitive = "text"
bind = { value = "first" }

[[panel]]
id = "trail"
primitive = "series"
bind = { y = "recent" }

[[panel]]
id = "world"
primitive = "scene"

[[panel.entity]]
path = "trail"
draw = "lines"
bind = { x = "recent.step", z = "recent.value" }

[[panel.entity]]
path = "flat"
draw = "surface"
bind = "act"
)";

}  // namespace

TEST_CASE("each panel shows the newest kept value at or before a held cursor") {
    const std::string ws = fresh_dir("resolve-cursor");
    StoreOpen opened = LoaderStore::open(ws);
    REQUIRE(opened.error.ok());
    LoaderStore& store = *opened.store;
    spec::ParseResult parsed = spec::parse_toml(kCursorSpec);
    INFO((parsed.diagnostics.empty() ? std::string() : parsed.diagnostics.front().message));
    REQUIRE(parsed.ok());
    const spec::Spec& s = parsed.spec;
    auto live = std::make_shared<LiveFake>();
    DataSources sources;
    sources.set(CatalogKind::live, live);
    StepHistory history;
    history.configure(s);
    for (int step : {10, 20, 30}) {
        history.append("live:act", step, row(float(step)));
        live->put(row(float(step)));
    }
    live->put(row(99));   // a newest value that no kept step holds yet

    ChainContext at_live{nullptr, &history, 30, true};
    PanelData d = resolve_panel(s, "now", store, {}, &sources, &at_live);
    REQUIRE(d.error.ok());
    CHECK(values_of(d.slots[0].tensor) == std::vector<double>{99, 100});
    CHECK_FALSE(d.slots[0].step);

    ChainContext held{nullptr, &history, 25, false};
    d = resolve_panel(s, "now", store, {}, &sources, &held);
    REQUIRE(d.error.ok());
    CHECK(values_of(d.slots[0].tensor) == std::vector<double>{20, 21});
    CHECK(d.slots[0].step == 20);
    // A source with `from` runs its own steps on the kept value.
    d = resolve_panel(s, "derived", store, {}, &sources, &held);
    REQUIRE(d.error.ok());
    CHECK(values_of(d.slots[0].tensor) == std::vector<double>{20});
    // panel.data says which step it shows.
    const json result = panel_data_result(s, resolve_panel(s, "now", store, {}, &sources, &held), 1);
    CHECK(result["data"]["cursor"]["step"] == 20);
    CHECK(result["data"]["cursor"]["live_only"] == false);

    // Before every kept step the panel shows its newest value: Live only.
    ChainContext early{nullptr, &history, 5, false};
    d = resolve_panel(s, "now", store, {}, &sources, &early);
    CHECK(values_of(d.slots[0].tensor) == std::vector<double>{99, 100});
    CHECK(d.slots[0].live_only);

    // The generation follows a held cursor, and panel_generation() agrees.
    const std::uint64_t g25 = resolve_panel(s, "now", store, {}, &sources, &held).generation;
    ChainContext held2{nullptr, &history, 10, false};
    const std::uint64_t g10 = resolve_panel(s, "now", store, {}, &sources, &held2).generation;
    CHECK(g25 != g10);
    CHECK(panel_generation(s, "now", sources, &held) == g25);
    CHECK(panel_generation(s, "now", sources, &at_live) == panel_generation(s, "now", sources));
}

TEST_CASE("a window stacks the kept steps up to the cursor with the step of each row") {
    const std::string ws = fresh_dir("resolve-window");
    StoreOpen opened = LoaderStore::open(ws);
    REQUIRE(opened.error.ok());
    spec::ParseResult parsed = spec::parse_toml(kCursorSpec);
    REQUIRE(parsed.ok());
    const spec::Spec& s = parsed.spec;
    auto live = std::make_shared<LiveFake>();
    DataSources sources;
    sources.set(CatalogKind::live, live);
    StepHistory history;
    history.configure(s);
    for (int step : {10, 20, 30}) {
        history.append("live:act", step, row(float(step)));
        live->put(row(float(step)));
    }
    ChainContext at_live{nullptr, &history, 30, true};
    PanelData d = resolve_panel(s, "trail", *opened.store, {}, &sources, &at_live);
    REQUIRE(d.error.ok());
    CHECK(d.slots[0].tensor.shape() == std::vector<std::int64_t>{2, 2});
    CHECK(values_of(d.slots[0].steps) == std::vector<double>{20, 30});

    ChainContext held{nullptr, &history, 20, false};
    d = resolve_panel(s, "trail", *opened.store, {}, &sources, &held);
    REQUIRE(d.error.ok());
    CHECK(values_of(d.slots[0].steps) == std::vector<double>{10, 20});
    CHECK(values_of(d.slots[0].tensor) == std::vector<double>{10, 11, 20, 21});

    // The entities of a scene panel bind the window's columns; a tensor
    // source without an operator gives no geometry.
    d = resolve_panel(s, "world", *opened.store, {}, &sources, &held);
    REQUIRE(d.error.ok());
    REQUIRE(d.entities.size() == 2);
    CHECK(d.entities[0].path == "trail");
    CHECK(d.entities[0].error.ok());
    REQUIRE(d.entities[0].slots.size() == 2);
    CHECK(d.entities[0].slots[0].slot == "x");
    CHECK(values_of(d.entities[0].slots[0].tensor) == std::vector<double>{10, 20});
    CHECK(d.entities[0].slots[1].slot == "z");
    CHECK(d.entities[0].slots[1].tensor.shape() == std::vector<std::int64_t>{2, 2});
    CHECK(d.entities[1].error.code == "entity.shape");
    CHECK(d.generation != 0);

    // A window with no kept step up to the cursor is not available yet.
    StepHistory empty;
    empty.configure(s);
    ChainContext none{nullptr, &empty, 0, true};
    CHECK(resolve_panel(s, "trail", *opened.store, {}, &sources, &none).error.code == "resolve.unavailable");
}

// A window whose steps read a state stops with resolve.out_of_range for a
// value outside 1 to 65,536, and never erases past the stacked steps.
TEST_CASE("a window whose steps read a state refuses a value outside the range of steps") {
    const std::string ws = fresh_dir("resolve-window-range");
    StoreOpen opened = LoaderStore::open(ws);
    REQUIRE(opened.error.ok());
    spec::ParseResult parsed = spec::parse_toml(R"(caliper_spec = 2
[instrument]
id = "steps"

[[state]]
id = "n"
type = "integer"

[[source]]
id = "act"
catalog = "live:act"

[[source]]
id = "recent"
from = "act"
transform = [{ op = "window", steps = { state = "n" } }]

[[panel]]
id = "trail"
primitive = "series"
bind = { y = "recent" }
)");
    INFO((parsed.diagnostics.empty() ? std::string() : parsed.diagnostics.front().message));
    REQUIRE(parsed.ok());
    const spec::Spec& s = parsed.spec;
    auto live = std::make_shared<LiveFake>();
    DataSources sources;
    sources.set(CatalogKind::live, live);
    StepHistory history;
    history.configure(s);
    for (int step : {10, 20, 30}) {
        history.append("live:act", step, row(float(step)));
        live->put(row(float(step)));
    }
    ValueStore values;
    values.reset(s);
    const ChainContext ctx{&values, &history, 30, true};

    REQUIRE(values.set_state("n", 2, "test", CommandOrigin::ui, nullptr).ok());
    PanelData d = resolve_panel(s, "trail", *opened.store, {}, &sources, &ctx);
    REQUIRE_MESSAGE(d.error.ok(), d.error.message);
    CHECK(values_of(d.slots[0].steps) == std::vector<double>{20, 30});
    for (const json bad : {json(-3), json(0), json(65537), json(std::numeric_limits<std::int64_t>::min())}) {
        INFO(bad.dump());
        REQUIRE(values.set_state("n", bad, "test", CommandOrigin::ui, nullptr).ok());
        d = resolve_panel(s, "trail", *opened.store, {}, &sources, &ctx);
        CHECK(d.error.code == "resolve.out_of_range");
        CHECK(d.error.where == "source recent, transform[0]");
    }
    REQUIRE(values.set_state("n", 1, "test", CommandOrigin::ui, nullptr).ok());
    d = resolve_panel(s, "trail", *opened.store, {}, &sources, &ctx);
    REQUIRE(d.error.ok());
    CHECK(values_of(d.slots[0].steps) == std::vector<double>{30});
}

// The arrays of one panel stop at 64 MiB, counted as float32 values in
// entity order before the app stages any: the entity that passes the limit
// has entity.limit, which names it, and the entities before it resolve.
TEST_CASE("the arrays of one panel stop at 64 MiB, at the entity that passes the limit") {
    const std::string ws = fresh_dir("resolve-panel-arrays");
    StoreOpen opened = LoaderStore::open(ws);
    REQUIRE(opened.error.ok());
    std::string text = "caliper_spec = 2\n[instrument]\nid = \"big\"\n[[source]]\nid = \"cloud\"\ncatalog = \"live:act\"\n"
                       "[[panel]]\nid = \"world\"\nprimitive = \"scene\"\n";
    for (int k = 1; k <= 6; ++k)
        text += "[[panel.entity]]\npath = \"e" + std::to_string(k) +
                "\"\ndraw = \"points\"\nbind = { positions = \"cloud\" }\n";
    spec::ParseResult parsed = spec::parse_toml(text);
    INFO((parsed.diagnostics.empty() ? std::string() : parsed.diagnostics.front().message));
    REQUIRE(parsed.ok());
    auto live = std::make_shared<LiveFake>();
    DataSources sources;
    sources.set(CatalogKind::live, live);
    // 1,048,576 points of 3 values: 12 MiB as float32, one block shared by the six entities.
    TensorResult cloud = TensorData::allocate(DType::f32, {spec::limits::kEntityItems, 3});
    REQUIRE(cloud.error.ok());
    std::memset(cloud.tensor.mutable_data(), 0, cloud.tensor.nbytes());
    live->put(cloud.tensor);

    const PanelData d = resolve_panel(parsed.spec, "world", *opened.store, {}, &sources);
    REQUIRE(d.error.ok());
    REQUIRE(d.entities.size() == 6);
    for (int k = 0; k < 5; ++k) {
        INFO(k);
        CHECK(d.entities[size_t(k)].error.ok());
        CHECK(d.entities[size_t(k)].slots.size() == 1);
    }
    CHECK(d.entities[5].error.code == "entity.limit");
    CHECK(d.entities[5].error.where == "e6");
    CHECK(d.entities[5].error.message.find("e6") != std::string::npos);
    CHECK(d.entities[5].error.message.find("world") != std::string::npos);
    CHECK(d.entities[5].slots.empty());
}
