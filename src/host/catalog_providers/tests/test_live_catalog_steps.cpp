// The tests of the live catalog's steps (design/instrument-interface.md,
// Step History): the step of each view's newest snapshot, from its stamp or
// the run's step at arrival, and the history that each kept snapshot feeds.
#include <doctest/doctest.h>

#include "catalog_providers/live_catalog.h"

#include <caliper/live_views.h>
#include <caliper_builder/session/events.h>
#include <caliper_builder/session/history.h>
#include <caliper_builder/spec/format.h>

#include <optional>
#include <string>
#include <vector>

using namespace caliper_host;
using namespace caliper_builder;

namespace {

struct Recorder : EventSink {
    std::vector<Event> events;
    std::uint64_t push(Event e) override {
        events.push_back(std::move(e));
        return events.size();
    }
};

LiveCatalog::ViewFrame frame(const std::string& name, std::uint64_t updates, const std::vector<float>& values,
                             std::optional<std::int64_t> step) {
    LiveCatalog::ViewFrame f;
    f.name = name;
    f.kind = CALIPER_LIVE_KIND_HEATMAP;
    f.updates = updates;
    f.has_payload = true;
    f.dtype = DType::f32;
    f.shape = {static_cast<std::int64_t>(values.size())};
    f.data = reinterpret_cast<const std::uint8_t*>(values.data());
    f.payload_bytes = values.size() * 4;
    f.step = step;
    return f;
}

struct Fixture {
    StepHistory history;
    Recorder events;
    std::optional<std::int64_t> run_step;
    std::shared_ptr<LiveCatalog> catalog;
    std::uint64_t clock = 1;
    Fixture(const std::string& depth = "") {
        spec::ParseResult r = spec::parse_toml("caliper_spec = 2\n[instrument]\nid = \"k\"\n[[source]]\nid = \"act\"\n"
                                               "catalog = \"live:act\"\n" + depth);
        REQUIRE(r.ok());
        history.configure(r.spec);
        LiveCatalog::Options o;
        o.history = &history;
        o.events = &events;
        o.run_step = [this] { return run_step; };
        o.clock = [this] { return clock; };
        catalog = std::make_shared<LiveCatalog>(o);
    }
};

}  // namespace

TEST_CASE("the live catalog keeps the step of each view's newest snapshot") {
    Fixture f;
    const std::vector<float> a{1, 2}, b{3, 4};
    CHECK_FALSE(f.catalog->step("live:act"));
    f.catalog->apply(true, "s1", {frame("act", 1, a, 5), frame("other", 1, b, 5)});
    CHECK(f.catalog->step("live:act") == 5);
    CHECK(f.catalog->step("act") == 5);
    // Without a stamp, the run's step when the snapshot arrived.
    f.run_step = 9;
    f.catalog->apply(true, "s1", {frame("act", 2, b, std::nullopt), frame("other", 1, b, 5)});
    CHECK(f.catalog->step("live:act") == 9);
    // Without either, no step and nothing kept.
    f.run_step.reset();
    f.catalog->apply(true, "s1", {frame("act", 3, a, std::nullopt), frame("other", 1, b, 5)});
    CHECK_FALSE(f.catalog->step("live:act"));
    CHECK(f.history.kept_steps() == std::vector<std::int64_t>{5, 9});
    const auto kept = f.history.at("act", 9);
    REQUIRE(kept);
    CHECK(kept->second.element(0) == doctest::Approx(3));
}

TEST_CASE("each snapshot of a kept view is copied, also while no reader wants it") {
    Fixture f;
    const std::vector<float> v{1};
    for (std::uint64_t u = 1; u <= 5; ++u) {
        f.catalog->apply(true, "s1", {frame("act", u, v, std::int64_t(u) * 10)});
        f.clock += 1000;   // well inside the idle interval, which would skip an unwanted view
    }
    CHECK(f.history.kept_steps() == std::vector<std::int64_t>{10, 20, 30, 40, 50});
    CHECK(f.catalog->copies() == 5);
}

TEST_CASE("a new attachment clears the kept steps, and a held cursor that falls out says clamped") {
    Fixture f("history = 2\n");
    const std::vector<float> v{1};
    f.catalog->apply(true, "s1", {frame("act", 1, v, 1)});
    f.catalog->apply(true, "s1", {frame("act", 2, v, 2)});
    CursorState hold = f.history.cursor();
    hold.live = false;
    hold.step = 1;
    f.history.set_cursor(hold);
    f.catalog->apply(true, "s1", {frame("act", 3, v, 3)});
    CHECK(f.history.cursor().step == 2);
    REQUIRE(f.events.events.size() == 1);
    CHECK(f.events.events[0].kind == EventKind::cursor);
    CHECK(f.events.events[0].fields["change"] == "clamped");
    CHECK(f.events.events[0].fields["step"] == 2);
    // A live cursor's moves add no event.
    hold.live = true;
    f.history.set_cursor(hold);
    f.catalog->apply(true, "s1", {frame("act", 4, v, 4)});
    CHECK(f.events.events.size() == 1);

    f.catalog->apply(true, "s2", {frame("act", 1, v, 1)});
    CHECK(f.history.kept_steps() == std::vector<std::int64_t>{1});
    f.catalog->apply(false, "", {});
    CHECK(f.history.kept_steps().empty());
}
