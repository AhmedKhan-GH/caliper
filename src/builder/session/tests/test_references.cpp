// The session tests of reference evaluation and the dependency index
// (design/instrument-interface.md, Re-Evaluation).
#include <doctest/doctest.h>

#include <caliper_builder/session/chain_hooks.h>
#include <caliper_builder/session/values.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/session/session.h>

#include <cstring>
#include <mutex>

#include "helpers.h"

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

// A data source that holds one served tensor per id.
class FakeSource : public CatalogDataSource {
public:
    std::uint64_t generation(const std::string& id) override {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = items.find(id);
        return it == items.end() ? 0 : it->second.generation;
    }
    ServedData read(const std::string& id, const ServedRequest&) override {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = items.find(id);
        if (it == items.end()) {
            ServedData gone;
            gone.error = Error("resolve.unavailable", "The source " + id + " went away.");
            return gone;
        }
        ServedData out = it->second.data;
        out.generation = it->second.generation;
        return out;
    }
    void put_tensor(const std::string& id, std::vector<int64_t> shape, const std::vector<float>& values) {
        std::vector<uint8_t> bytes(values.size() * 4);
        std::memcpy(bytes.data(), values.data(), bytes.size());
        TensorResult t = TensorData::from_bytes(DType::f32, std::move(shape), std::move(bytes));
        REQUIRE(t.error.ok());
        std::lock_guard<std::mutex> lock(mutex);
        Item& item = items[id];
        ++item.generation;
        item.data = ServedData{};
        item.data.tensor = t.tensor;
    }

    struct Item {
        std::uint64_t generation = 0;
        ServedData data;
    };
    std::mutex mutex;
    std::map<std::string, Item> items;
};

spec::Spec parse(const std::string& body) {
    auto parsed = spec::parse_toml("caliper_spec = 2\n[instrument]\nid = \"a\"\n" + body);
    REQUIRE_MESSAGE(parsed.ok(), body);
    return parsed.spec;
}

// Two sources of live tensors (10, 3): `sa` slices row k, `sb` slices row 2.
const char* kEvalSpec = R"(
[[source]]
id = "sa"
catalog = "live:a"
  [[source.transform]]
  op = "slice"
  axis = 0
  index = { control = "k" }

[[source]]
id = "sb"
catalog = "live:b"
  [[source.transform]]
  op = "slice"
  axis = 0
  index = 2

[[control]]
id = "k"
kind = "slider"
bind = "value"
options = { min = 0, max = 9 }

[[control]]
id = "other"
kind = "slider"
bind = "value"
options = { min = 0, max = 9 }

[[panel]]
id = "pa"
primitive = "histogram"
bind = { values = "sa" }

[[panel]]
id = "pb"
primitive = "histogram"
bind = { values = "sb" }
)";

// A slice step whose index reads the control k, with the given default text.
spec::Spec step_spec(const std::string& reference) {
    return parse("[[source]]\nid = \"s\"\ncatalog = \"live:a\"\n  [[source.transform]]\n  op = \"slice\"\n  axis = 0\n"
                 "  index = " + reference + "\n[[control]]\nid = \"k\"\nkind = \"slider\"\nbind = \"value\"\n"
                 "options = { min = 0, max = 9 }\n");
}

std::vector<float> ramp(int n) {
    std::vector<float> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = static_cast<float>(i);
    return v;
}

}  // namespace

// A panel whose chain reads a control computes at most once a frame, and a chain without references once in all.
TEST_CASE("references: evaluation count over 100 control values in 10 frames") {
    auto s = open_session(fresh_dir("references-eval") + "/workspace");
    auto fake = std::make_shared<FakeSource>();
    fake->put_tensor("live:a", {10, 3}, ramp(30));
    fake->put_tensor("live:b", {10, 3}, ramp(30));
    DataSources sources;
    REQUIRE(sources.set(CatalogKind::live, fake));

    const spec::Spec sp = parse(kEvalSpec);
    ValueStore values;
    values.reset(sp);
    const ChainContext chain{&values, nullptr, 0, true};

    std::uint64_t computed_a = 0, computed_b = 0;
    for (int frame = 0; frame < 10; ++frame) {
        for (int i = 0; i < 10; ++i)
            REQUIRE(values.set_control("k", (frame + i) % 10, CommandOrigin::ui, nullptr).ok());
        const auto before_a = sources.cache_stats().computed;
        PanelData a = resolve_panel(sp, "pa", s->store(), {}, &sources, &chain);
        REQUIRE_MESSAGE(a.error.ok(), a.error.message);
        const auto after_a = sources.cache_stats().computed;
        PanelData b = resolve_panel(sp, "pb", s->store(), {}, &sources, &chain);
        REQUIRE_MESSAGE(b.error.ok(), b.error.message);
        const auto after_b = sources.cache_stats().computed;
        CAPTURE(frame);
        CHECK(after_a - before_a == 1u);   // the last value of the frame differs from the one before
        CHECK(after_b - after_a == (frame == 0 ? 1u : 0u));
        computed_a += after_a - before_a;
        computed_b += after_b - after_a;
    }
    CHECK(computed_a == 10);   // the last value of each frame differs from the one before, so each frame evaluates once
    CHECK(computed_b == 1);
}

// A reference in a step gives the slider's value rounded, its default without a store, or resolve.waiting.
TEST_CASE("references: substitute_references") {
    const spec::Spec sp = step_spec("{ control = \"k\" }");
    const spec::Transform& step = sp.sources[0].transforms[0];
    REQUIRE(step.references.count("index") == 1);

    ValueStore values;
    values.reset(sp);
    const ChainContext with_store{&values, nullptr, 0, true};
    REQUIRE(values.set_control("k", 3.6, CommandOrigin::ui, nullptr).ok());
    spec::Transform out;
    CHECK(substitute_references(step, with_store, &out).ok());
    CHECK(out.index == std::optional<std::int64_t>(4));
    CHECK(out.references.empty());
    REQUIRE(values.set_control("k", 3.2, CommandOrigin::ui, nullptr).ok());
    CHECK(substitute_references(step, with_store, &out).ok());
    CHECK(out.index == std::optional<std::int64_t>(3));

    // With no store, the default is the value.
    const spec::Spec with_default = step_spec("{ control = \"k\", default = 5 }");
    const ChainContext no_store{nullptr, nullptr, 0, true};
    CHECK(substitute_references(with_default.sources[0].transforms[0], no_store, &out).ok());
    CHECK(out.index == std::optional<std::int64_t>(5));

    // With no value and no default, the step waits.
    Error waiting = substitute_references(step, no_store, &out);
    CHECK(waiting.code == "resolve.waiting");
}

// Each value that a reference substitutes is checked against the range of its
// place, the range the validator applies to a literal there: the window's
// steps from 1 to 65,536, a slice's step from 1, and every place within
// +-(2^53 - 1), with a value that does not round exactly refused.
TEST_CASE("references: a value outside the range of its place is refused at the substitution") {
    const spec::Spec sp = parse(R"(
[[state]]
id = "n"
type = "number"

[[source]]
id = "act"
catalog = "live:a"

[[source]]
id = "recent"
from = "act"
  [[source.transform]]
  op = "window"
  steps = { state = "n" }

[[source]]
id = "cut"
catalog = "live:b"
  [[source.transform]]
  op = "slice"
  axis = 0
  start = { state = "n" }
  step = { state = "n" }
)");
    const spec::Transform& window = sp.sources[1].transforms[0];
    const spec::Transform& slice = sp.sources[2].transforms[0];
    ValueStore values;
    values.reset(sp);
    const ChainContext ctx{&values, nullptr, 0, true};
    spec::Transform out;

    REQUIRE(values.set_state("n", 2.6, "test", CommandOrigin::ui, nullptr).ok());
    CHECK(substitute_references(window, ctx, &out).ok());
    CHECK(out.steps == std::optional<std::int64_t>(3));
    CHECK(substitute_references(slice, ctx, &out).ok());
    CHECK(out.step == std::optional<std::int64_t>(3));
    REQUIRE(values.set_state("n", 65536, "test", CommandOrigin::ui, nullptr).ok());
    CHECK(substitute_references(window, ctx, &out).ok());

    for (const double bad : {-3.0, 0.0, 0.4, 65537.0, 1e300, 9007199254740992.0, -1e19}) {
        INFO(bad);
        REQUIRE(values.set_state("n", bad, "test", CommandOrigin::ui, nullptr).ok());
        const Error e = substitute_references(window, ctx, &out);
        CHECK(e.code == "resolve.out_of_range");
        CHECK(e.where == "steps");
        CHECK(e.message.find("from 1 to 65536") != std::string::npos);
    }
    // A slice's step is at least 1; its start may be negative, within 2^53 - 1.
    for (const double bad : {0.0, -1.0, 1e300, 9007199254740992.0}) {
        INFO(bad);
        REQUIRE(values.set_state("n", bad, "test", CommandOrigin::ui, nullptr).ok());
        CHECK(substitute_references(slice, ctx, &out).code == "resolve.out_of_range");
    }
    // The table holds every key that the validator lets a reference take.
    for (const spec::IntegerPlace& place : spec::integer_places()) {
        INFO(place.key);
        CHECK(place.min <= place.max);
        CHECK(spec::integer_place(place.op, place.key) == &place);
    }
    CHECK(spec::integer_place(spec::TransformOp::window, "index") == nullptr);
    CHECK(spec::integer_place(spec::TransformOp::reduce, "axis") == nullptr);
}

// The key holds the values that the chain reads, so it is empty for a chain without references and moves with them.
TEST_CASE("references: references_key") {
    const spec::Spec sp = parse(kEvalSpec);
    ValueStore values;
    values.reset(sp);
    const ChainContext chain{&values, nullptr, 0, true};
    const std::vector<const spec::Source*> reads = {&sp.sources[0]};
    const std::vector<const spec::Source*> plain = {&sp.sources[1]};

    CHECK(references_key(plain, chain).empty());
    REQUIRE(values.set_control("k", 1, CommandOrigin::ui, nullptr).ok());
    const std::string one = references_key(reads, chain);
    CHECK_FALSE(one.empty());
    REQUIRE(values.set_control("k", 2, CommandOrigin::ui, nullptr).ok());
    const std::string two = references_key(reads, chain);
    CHECK(two != one);
    REQUIRE(values.set_control("other", 7, CommandOrigin::ui, nullptr).ok());
    CHECK(references_key(reads, chain) == two);
}

// The index lists each reader of an input, and finds the cycle of a control that reads its own value.
TEST_CASE("references: dependency index") {
    const spec::Spec sp = parse(R"(
[[state]]
id = "token"
type = "integer"
min = 0
max = 63

[[state]]
id = "w"
type = "range"
min = 0
max = 10

[[source]]
id = "s1"
catalog = "live:x"
  [[source.transform]]
  op = "slice"
  axis = 0
  index = { state = "token" }

[[panel]]
id = "h"
primitive = "histogram"
bind = { values = "s1" }

[[control]]
id = "win"
kind = "range"
bind = "state:w"
)");
    DependencyIndex index;
    index.rebuild(sp);
    auto paths = [&](InputKey::Kind kind, const std::string& id) {
        std::vector<std::string> out;
        for (const auto& r : index.readers_of(InputKey{kind, id})) out.push_back(r.path);
        return out;
    };
    auto holds = [](const std::vector<std::string>& list, const std::string& path) {
        return std::find(list.begin(), list.end(), path) != list.end();
    };
    CHECK(holds(paths(InputKey::Kind::state, "token"), "source[0].transform[0]"));
    CHECK(holds(paths(InputKey::Kind::state, "w"), "control[0]"));
    CHECK(holds(paths(InputKey::Kind::source, "s1"), "panel[0].bind.values"));
    CHECK(index.cycles().empty());

    const spec::Spec self = parse(R"(
[[control]]
id = "window"
kind = "range"
bind = "value"
options = { min = { control = "window", field = "lo" }, max = 10 }
)");
    DependencyIndex cyclic;
    cyclic.rebuild(self);
    const auto cycles = cyclic.cycles();
    REQUIRE(cycles.size() >= 1);
    CHECK(cycles[0].code == "spec.reference_cycle");
}
