// The session tests of the kept steps (design/instrument-interface.md, Step
// History): the depth, the history point, one value per step, the budget,
// the reset, and the cursor's moves over the kept steps.
#include <doctest/doctest.h>

#include <caliper_builder/session/history.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/spec/format.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
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

const char* kHead = "caliper_spec = 2\n[instrument]\nid = \"h\"\n";

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
    h.append("live:act", 30, filled({1}, 31));   // a later value of the newest step replaces it
    CHECK(h.kept_steps() == std::vector<std::int64_t>{10, 20, 30});
    CHECK(h.at("act", 25)->first == 20);
    CHECK(h.at("act", 25)->second.element(0) == doctest::Approx(20));
    CHECK(h.at("act", 35)->second.element(0) == doctest::Approx(31));
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
    // Every view shares this 64 KiB buffer. The history counts each retained
    // value, but the test allocates no large payload. Each large ring asks
    // for 125 MiB, below the per-source cap. Add enough rings to exceed the
    // total budget while the small ring retains all of its steps.
    const TensorData value = filled({16384}, 1);
    constexpr std::int64_t large_steps = 2000, small_steps = 64;
    const std::uint64_t step_bytes = value.nbytes();
    const std::uint64_t small_bytes = small_steps * step_bytes;
    const std::uint64_t large_bytes = large_steps * step_bytes;
    REQUIRE(large_bytes < kHistorySourceBytes);
    const std::size_t rings = (kHistoryBudgetBytes - small_bytes) / large_bytes + 1;
    REQUIRE(rings * large_bytes + small_bytes > kHistoryBudgetBytes);
    std::string toml = std::string(kHead) + "[[source]]\nid = \"small\"\ncatalog = \"live:b\"\nhistory = 64\n";
    for (std::size_t i = 1; i <= rings; ++i)
        toml += "[[source]]\nid = \"big" + std::to_string(i) + "\"\ncatalog = \"live:a\"\nhistory = 2000\n";
    const spec::Spec s = parse(toml);
    StepHistory h;
    h.configure(s);
    for (std::int64_t step = 0; step < large_steps; ++step) {
        h.append("live:a", step, value);
        h.append("live:b", step, value);
    }
    std::map<std::string, KeptSource> kept;
    std::uint64_t total = 0;
    for (const KeptSource& k : h.kept()) {
        kept[k.id] = k;
        total += k.bytes;
    }
    REQUIRE(kept.size() == rings + 1);
    CHECK(total <= kHistoryBudgetBytes);
    CHECK(kHistoryBudgetBytes - total < rings * step_bytes);
    CHECK(kept["small"].steps == small_steps);
    CHECK(kept["small"].note.empty());
    const auto expected = static_cast<std::int64_t>((kHistoryBudgetBytes - small_bytes) / rings / step_bytes);
    REQUIRE(expected < large_steps);
    for (std::size_t i = 1; i <= rings; ++i) {
        const std::string id = "big" + std::to_string(i);
        CAPTURE(id);
        CHECK(kept[id].requested == large_steps);
        CHECK(kept[id].steps == expected);
        CHECK(kept[id].last == large_steps - 1);
        CHECK(kept[id].note.find(id) != std::string::npos);
    }
}

TEST_CASE("one raw source keeps at most 128 MiB and says which step keeps more") {
    // A raw activation uses 8 MiB per step. Its default 64 steps request
    // 512 MiB, above the per-source cap. The views share one buffer, and
    // the reduced source makes 16 KiB values.
    const TensorData value = filled({512, 4096}, 1);
    const spec::Spec s = parse(std::string(kHead) +
                               "[[source]]\nid = \"raw\"\ncatalog = \"live:mlp.act\"\n"
                               "[[source]]\nid = \"mean\"\ncatalog = \"live:mlp.act\"\n"
                               "[[source.transform]]\nop = \"reduce\"\nfn = \"mean\"\naxis = 0\n");
    StepHistory h;
    h.configure(s);
    for (int step = 1; step <= 64; ++step) h.append("live:mlp.act", step, value);
    std::map<std::string, KeptSource> kept;
    for (const KeptSource& k : h.kept()) kept[k.id] = k;
    REQUIRE(kept.size() == 2);
    CHECK(kHistorySourceBytes == std::uint64_t(128) << 20);
    CHECK(kept["raw"].requested == kDefaultLiveDepth);
    CHECK(kept["raw"].steps == (std::int64_t)(kHistorySourceBytes / value.nbytes()));
    CHECK(kept["raw"].steps == 16);
    CHECK(kept["raw"].bytes <= kHistorySourceBytes);
    CHECK(kept["raw"].last == 64);
    CHECK(kept["raw"].note.find("The source raw keeps 16 of its 64 steps") == 0);
    CHECK(kept["raw"].note.find("reduce step") != std::string::npos);
    CHECK(kept["mean"].steps == 64);
    CHECK(kept["mean"].note.empty());
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

TEST_CASE("a live series keeps its newest value at each step only for a source that states history") {
    const spec::Spec s = parse(std::string(kHead) +
                               "[[source]]\nid = \"loss\"\ncatalog = \"live:loss\"\nhistory = 2048\n"
                               "[[source]]\nid = \"plain\"\ncatalog = \"live:loss\"\n");
    StepHistory h;
    h.configure(s);
    for (int step = 1; step <= 5; ++step) h.append_series("live:loss", step * 10, 1.0 / step);
    std::map<std::string, KeptSource> kept;
    for (const KeptSource& k : h.kept()) kept[k.id] = k;
    CHECK(kept["loss"].steps == 5);
    CHECK(kept["loss"].first == 10);
    CHECK(kept["loss"].last == 50);
    CHECK(kept["plain"].steps == 0);
    const auto at = h.at("loss", 35);
    REQUIRE(at);
    CHECK(at->first == 30);
    CHECK(at->second.rank() == 0);
    CHECK(at->second.element(0) == doctest::Approx(1.0 / 3));
}

// A step counter that restarts within one attachment (a new run on the same
// link, or a fork from an earlier step) starts a new epoch: the old run's
// steps after the new step go, so that a full ring keeps the new values, and
// a cursor held after the new step goes live.
TEST_CASE("a step counter that restarts starts a new epoch and keeps the new values") {
    const spec::Spec s = parse(std::string(kHead) +
                               "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\nhistory = 3\n"
                               "[[source]]\nid = \"other\"\ncatalog = \"live:other\"\nhistory = 3\n");
    StepHistory h;
    h.configure(s);
    for (int step : {100, 101, 102, 103}) {
        h.append("live:act", step, filled({1}, float(step)));
        h.append("live:other", step, filled({1}, float(step)));
    }
    CHECK(h.kept_steps() == std::vector<std::int64_t>{101, 102, 103});
    CursorState held = h.cursor();
    held.live = false;
    held.step = 102;
    h.set_cursor(held);
    REQUIRE_FALSE(h.cursor().live);

    // The new run counts from 0: its values are kept, and the old steps go.
    for (int step : {0, 1}) h.append("live:act", step, filled({1}, float(step)));
    CHECK(h.kept_steps() == std::vector<std::int64_t>{0, 1});
    CHECK(h.at("act", 1)->second.element(0) == doctest::Approx(1));
    CHECK_FALSE(h.at("other", 103));
    CursorState c = h.cursor();
    CHECK(c.live);
    CHECK(c.latest == 1);
    CHECK(c.first == 0);
    CHECK(c.step == 1);
    // The other view of the new run continues in the same epoch.
    h.append("live:other", 1, filled({1}, 1));
    for (int step : {2, 3, 4, 5}) h.append("live:act", step, filled({1}, float(step)));
    CHECK(h.window("act", 0, 10).size() == 3);
    CHECK(h.window("act", 0, 10).front().first == 3);
    CHECK(h.at("other", 1).has_value());
    CHECK(h.cursor().latest == 5);
    // A step that repeats the newest is a later value of that step, not a restart.
    h.append("live:act", 5, filled({1}, 50));
    CHECK(h.kept_steps().size() == 4);
}

// A fork from step 30 of a run at step 50 keeps the shared steps up to 30 and
// a cursor held there, and drops the old run's steps after it.
TEST_CASE("a fork from an earlier step keeps the shared steps and a cursor held on them") {
    const spec::Spec s = parse(std::string(kHead) + "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\nhistory = 8\n");
    StepHistory h;
    h.configure(s);
    for (int step : {10, 20, 30, 40, 50}) h.append("live:act", step, filled({1}, float(step)));
    CursorState held = h.cursor();
    held.live = false;
    held.step = 20;
    h.set_cursor(held);
    h.append("live:act", 30, filled({1}, 300));
    CHECK(h.kept_steps() == std::vector<std::int64_t>{10, 20, 30});
    CHECK(h.at("act", 35)->second.element(0) == doctest::Approx(300));
    CHECK_FALSE(h.cursor().live);
    CHECK(h.cursor().step == 20);
    CHECK(h.cursor().latest == 30);
    h.append("live:act", 31, filled({1}, 31));
    CHECK(h.kept_steps() == std::vector<std::int64_t>{10, 20, 30, 31});
}

// The nearest kept step of a held cursor compares distances that do not fit a
// signed 64-bit value: the step 100 between the stamps -2^63 + 10 and
// 2^63 - 10 is nearer the upper one.
TEST_CASE("the nearest kept step is exact for stamps far apart") {
    const spec::Spec s = parse(std::string(kHead) + "[[source]]\nid = \"act\"\ncatalog = \"live:act\"\nhistory = 3\n");
    StepHistory h;
    h.configure(s);
    const std::int64_t low = std::numeric_limits<std::int64_t>::min() + 10;
    const std::int64_t high = std::numeric_limits<std::int64_t>::max() - 10;
    h.append("live:act", low, filled({1}, 1));
    h.append("live:act", high, filled({1}, 2));
    CursorState want = h.cursor();
    want.live = false;
    want.step = 100;
    CHECK(h.set_cursor(want).step == high);
    want.step = -100;
    CHECK(h.set_cursor(want).step == low);
}


TEST_CASE("resource history leaves 256 MiB for sources at the full piece budget") {
    CHECK(kHistoryBudgetBytes == std::uint64_t(768) << 20);
    CHECK(kHistoryResourceBytes == std::uint64_t(512) << 20);
    CHECK(kHistoryBudgetBytes - kHistoryResourceBytes == std::uint64_t(256) << 20);
    CHECK(kHistorySourceBytes == std::uint64_t(128) << 20);
}
