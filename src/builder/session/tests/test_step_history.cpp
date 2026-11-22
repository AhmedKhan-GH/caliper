// The session tests of the kept steps (design/instrument-interface.md, Step
// History): the depth, the history point, one value per step, the budget,
// the reset, and the cursor's moves over the kept steps.
#include <doctest/doctest.h>

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/format.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace caliper_builder;

namespace {

spec::Spec parse(const std::string& toml) {
    spec::ParseResult r = spec::parse_toml(toml);
    INFO(toml);
    REQUIRE(r.ok());
    return r.spec;
}

// A f32 tensor of `shape`, every element `fill`; a fresh buffer each call.
TensorData filled(std::vector<std::int64_t> shape, float fill) {
    TensorResult r = TensorData::allocate(DType::f32, shape);
    REQUIRE(r.error.ok());
    auto* p = reinterpret_cast<float*>(r.tensor.mutable_data());
    for (std::uint64_t i = 0; i < r.tensor.numel(); ++i) p[i] = fill;
    return r.tensor;
}

const char* kHead = "caliper_spec = 1\n[applet]\nid = \"h\"\n";

}  // namespace

TEST_CASE("a live tensor source keeps 64 steps by default and its history when given") {
    const spec::Spec s = parse(std::string(kHead) +
                               "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n"
                               "[[source]]\nid = \"deep\"\ncatalog = \"live:act\"\nhistory = 100\n"
                               "[[source]]\nid = \"none\"\ncatalog = \"live:act\"\nhistory = 0\n"
                               "[[source]]\nid = \"derived\"\nfrom = \"act\"\n"
                               "[[source]]\nid = \"file\"\ncatalog = \"tensor:w\"\n");
    CHECK(history_depth(s, *s.find_source("act")) == kDefaultLiveDepth);
    CHECK(history_depth(s, *s.find_source("deep")) == 100);
    CHECK(history_depth(s, *s.find_source("none")) == 0);
    CHECK(history_depth(s, *s.find_source("derived")) == 0);
    CHECK(history_depth(s, *s.find_source("file")) == 0);

    StepHistory h;
    h.configure(s);
    for (int step = 1; step <= 150; ++step) h.append("live:act", step, filled({4}, float(step)));
    std::map<std::string, KeptSource> kept;
    for (const KeptSource& k : h.kept()) kept[k.id] = k;
    REQUIRE(kept.size() == 2);
    CHECK(kept["act"].steps == 64);
    CHECK(kept["act"].first == 87);
    CHECK(kept["act"].last == 150);
    CHECK(kept["act"].requested == 64);
    CHECK(kept["act"].bytes == 64 * 16);
    CHECK(kept["deep"].steps == 100);
    CHECK(kept["deep"].first == 51);
    const std::vector<std::string> ids = kept_catalog_ids(h);
    CHECK(ids == std::vector<std::string>{"live:act"});
}

TEST_CASE("the kept value is the value after the longest reference-free run of shipped steps") {
    const spec::Spec s = parse(std::string(kHead) +
                               "[[state]]\nid = \"token\"\ntype = \"integer\"\n"
                               "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n"
                               "transform = [{ op = \"reduce\", fn = \"sum\", axis = 1 },"
                               " { op = \"slice\", axis = 0, index = { state = \"token\" } }]\n");
    const spec::Source& act = *s.find_source("act");
    CHECK(history_point(s, act) == 1);
    StepHistory h;
    h.configure(s);
    h.append("live:act", 7, filled({3, 2}, 1.5f));
    const auto v = h.at("act", 7);
    REQUIRE(v);
    CHECK(v->first == 7);
    CHECK(v->second.shape() == std::vector<std::int64_t>{3});
    CHECK(v->second.element(0) == doctest::Approx(3.0));
}

TEST_CASE("one value per step, the newest at or before a step, and a window by step numbers") {
    const spec::Spec s = parse(std::string(kHead) + "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n");
    StepHistory h;
    h.configure(s);
    h.append("live:other", 3, filled({1}, 9));   // not a root of the spec: nothing kept
    CHECK(h.kept_steps().empty());
    for (int step : {10, 20, 30}) h.append("live:act", step, filled({1}, float(step)));
    h.append("live:act", 20, filled({1}, 21));   // a later value of a step replaces the earlier one
    CHECK(h.kept_steps() == std::vector<std::int64_t>{10, 20, 30});
    CHECK(h.at("act", 25)->first == 20);
    CHECK(h.at("act", 25)->second.element(0) == doctest::Approx(21));
    CHECK_FALSE(h.at("act", 5));
    const auto w = h.window("act", 15, 30);
    REQUIRE(w.size() == 2);
    CHECK(w[0].first == 20);
    CHECK(w[1].first == 30);
    // A new shape starts the ring over.
    h.append("live:act", 40, filled({2}, 1));
    CHECK(h.kept_steps() == std::vector<std::int64_t>{40});
}

TEST_CASE("a change of a source's chain clears its steps, and an unrelated edit keeps them") {
    const std::string base = std::string(kHead) + "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n";
    StepHistory h;
    h.configure(parse(base));
    h.append("live:act", 1, filled({2}, 1));
    h.configure(parse(base + "[[source]]\nid = \"other\"\ncatalog = \"live:b\"\n"));
    CHECK(h.kept_steps() == std::vector<std::int64_t>{1});
    h.configure(parse(std::string(kHead) + "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n"
                                           "transform = [{ op = \"reduce\", fn = \"mean\", axis = 0 }]\n"));
    CHECK(h.kept_steps().empty());
    h.append("live:act", 2, filled({2}, 1));
    h.clear();
    CHECK(h.kept_steps().empty());
    CHECK(h.cursor().live);
}

TEST_CASE("the budget shortens the largest rings first") {
    // 64 KiB values; the views share one buffer, so the test holds little
    // memory while the history counts every value's bytes.
    const TensorData value = filled({16384}, 1);
    const spec::Spec s = parse(std::string(kHead) +
                               "[[source]]\nid = \"big\"\ncatalog = \"live:a\"\nhistory = 9000\n"
                               "[[source]]\nid = \"small\"\ncatalog = \"live:b\"\nhistory = 64\n");
    StepHistory h;
    h.configure(s);
    for (int step = 0; step < 9000; ++step) {
        h.append("live:a", step, value);
        h.append("live:b", step, value);
    }
    std::map<std::string, KeptSource> kept;
    std::uint64_t total = 0;
    for (const KeptSource& k : h.kept()) {
        kept[k.id] = k;
        total += k.bytes;
    }
    CHECK(total <= kHistoryBudgetBytes);
    CHECK(kept["small"].steps == 64);
    CHECK(kept["big"].requested == 9000);
    CHECK(kept["big"].steps == (std::int64_t)((kHistoryBudgetBytes - 64 * 65536) / 65536));
    CHECK(kept["big"].last == 8999);
}

TEST_CASE("the cursor is live, holds at the nearest kept step, steps, plays and clamps") {
    const spec::Spec s = parse(std::string(kHead) + "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\nhistory = 3\n"
                                                    "[[control]]\nid = \"t\"\nkind = \"transport\"\nbind = \"cursor\"\n"
                                                    "options = { speed = 20 }\n");
    StepHistory h;
    h.configure(s);
    CHECK(h.cursor().live);
    CHECK(h.cursor().speed == doctest::Approx(20));
    for (int step : {10, 20, 30}) h.append("live:act", step, filled({1}, float(step)));
    CursorState c = h.cursor();
    CHECK(c.live);
    CHECK(c.step == 30);
    CHECK(c.first == 10);
    CHECK(c.latest == 30);

    CursorState want = c;
    want.live = false;
    want.step = 14;
    c = h.set_cursor(want);
    CHECK_FALSE(c.live);
    CHECK(c.step == 10);   // the nearest kept step
    c = h.step_cursor(1);
    CHECK(c.step == 20);
    c = h.step_cursor(-5);
    CHECK(c.step == 10);
    c = h.step_cursor(100);
    CHECK(c.step == 30);
    CHECK_FALSE(c.live);

    // A held cursor stays while the run goes on, until its step falls out.
    want = h.cursor();
    want.step = 10;
    h.set_cursor(want);
    h.append("live:act", 40, filled({1}, 40));
    CHECK(h.cursor().step == 20);   // 10 fell out: the cursor moved to first
    CHECK(h.cursor().latest == 40);

    // Play moves at the speed and goes live at the newest step.
    want = h.cursor();
    want.playing = true;
    c = h.set_cursor(want);
    CHECK(c.playing);
    h.tick(0.5);   // 10 steps at 20 a second
    CHECK(h.cursor().step == 30);
    h.tick(1.0);
    CHECK(h.cursor().live);
    CHECK(h.cursor().step == 40);
    CHECK_FALSE(h.cursor().playing);

    want = h.cursor();
    want.live = true;
    CHECK(h.set_cursor(want).step == 40);
}

TEST_CASE("history memory of the default depth" * doctest::skip(std::getenv("CALIPER_HISTORY_MEASURE") == nullptr)) {
    // Measured under /usr/bin/time -l: the default 64 steps of one raw
    // hidden activation of the small (256, 1024) and the medium (512, 4096)
    // multilayer perceptron, each value a fresh copy as the live catalog
    // makes one per snapshot.
    const std::string size = std::getenv("CALIPER_HISTORY_MEASURE");
    const std::vector<std::int64_t> shape =
        size == "medium" ? std::vector<std::int64_t>{512, 4096} : std::vector<std::int64_t>{256, 1024};
    const spec::Spec s = parse(std::string(kHead) + "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\n");
    StepHistory h;
    h.configure(s);
    for (int step = 0; step < 80; ++step) h.append("live:act", step, filled(shape, float(step)));
    const KeptSource k = h.kept().front();
    MESSAGE("history " << size << ": " << k.steps << " steps, " << k.bytes << " bytes");
    CHECK(k.steps == kDefaultLiveDepth);
}
