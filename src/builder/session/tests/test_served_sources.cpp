// Tests of the sources the app serves, through the seam the session gives it:
// a CatalogDataSource registered per kind in DataSources, a live tensor with
// the spec's transforms on it (entropy and mean, checked by hand), the
// generation of a panel and the cheap probe that predicts it, the cache that
// runs a chain's transforms once per generation, columns of a served table,
// a source that went away, a tensor past the limit, and the registry's rules.
#include <caliper_builder/catalog/memory_provider.h>
#include <caliper_builder/session/resolve.h>
#include <caliper_builder/session/session.h>

#include <doctest/doctest.h>

#include "helpers.h"

#include <cmath>
#include <cstring>
#include <map>
#include <mutex>

using namespace caliper_builder;
using namespace session_test;
using nlohmann::json;

namespace {

// A data source that holds one tensor or one table per id, with a generation
// that the test bumps, and counts its reads.
class FakeSource : public CatalogDataSource {
public:
    struct Item {
        std::uint64_t generation = 0;
        ServedData data;
    };
    std::uint64_t generation(const std::string& id) override {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = items.find(id);
        return it == items.end() ? 0 : it->second.generation;
    }
    ServedData read(const std::string& id, const ServedRequest& request) override {
        std::lock_guard<std::mutex> lock(mutex);
        ++reads;
        last_request = request;
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
    void put_table(const std::string& id, std::vector<ColumnInfo> columns, json rows) {
        std::lock_guard<std::mutex> lock(mutex);
        Item& item = items[id];
        ++item.generation;
        item.data = ServedData{};
        item.data.tabular = true;
        item.data.columns = std::move(columns);
        item.data.rows = std::move(rows);
    }
    void erase(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex);
        items.erase(id);
    }

    std::mutex mutex;
    std::map<std::string, Item> items;
    int reads = 0;
    ServedRequest last_request;
};

// [layers=2, heads=2, tokens=2, tokens=3]: the rows of the last axis.
const std::vector<float> kAttention = {
    1, 1, 1,  1, 2, 1,    // layer 0, head 0
    4, 0, 0,  1, 1, 2,    // layer 0, head 1
    1, 3, 0,  2, 2, 2,    // layer 1, head 0
    0, 0, 5,  3, 1, 0,    // layer 1, head 1
};

// The entropy in nats of one row, dividing by its sum and skipping values <= 0.
double entropy(std::vector<double> row) {
    double sum = 0;
    for (double v : row) sum += v > 0 ? v : 0;
    double h = 0;
    for (double v : row)
        if (v > 0) h -= (v / sum) * std::log(v / sum);
    return h;
}

// Entropy over the last axis, then the mean over the query tokens: [layers, heads].
std::vector<double> expected_entropy(const std::vector<float>& a) {
    std::vector<double> out;
    for (int lh = 0; lh < 4; ++lh) {
        double m = 0;
        for (int q = 0; q < 2; ++q) {
            const float* r = &a[static_cast<std::size_t>(lh * 6 + q * 3)];
            m += entropy({r[0], r[1], r[2]});
        }
        out.push_back(m / 2);
    }
    return out;
}

std::vector<double> values_of(const TensorData& t) {
    std::vector<double> out;
    for (uint64_t i = 0; i < t.numel(); ++i) out.push_back(t.element(i));
    return out;
}

CatalogEntry live_entry(const std::string& name, std::vector<int64_t> shape) {
    CatalogEntry e;
    e.kind = CatalogKind::live;
    e.name = name;
    e.id = make_catalog_id(e.kind, name);
    e.title = name;
    e.dtype = CatalogDtype::f32;
    e.shape = std::move(shape);
    e.wire = true;
    return e;
}

CatalogEntry feed_entry(const std::string& name) {
    CatalogEntry e;
    e.kind = CatalogKind::feed;
    e.name = name;
    e.id = make_catalog_id(e.kind, name);
    e.title = name;
    e.columns = std::vector<CatalogColumn>{{"seq", "BIGINT"}, {"t_ns", "BIGINT"}, {"value", "FLOAT"}};
    e.rows = 0;
    return e;
}

struct Fixture {
    std::unique_ptr<Session> s;
    std::shared_ptr<FakeSource> live = std::make_shared<FakeSource>();
    std::shared_ptr<FakeSource> feed = std::make_shared<FakeSource>();
    explicit Fixture(const std::string& name) {
        s = open_session(fresh_dir(name) + "/workspace");
        auto entries = std::make_shared<InMemoryCatalogProvider>();
        REQUIRE(entries->put(live_entry("attn.weights", {2, 2, 2, 3})).ok());
        REQUIRE(entries->put(feed_entry("sys.cpu.util")).ok());
        REQUIRE(s->catalog().add_provider("served", entries).ok());
        live->put_tensor("live:attn.weights", {2, 2, 2, 3}, kAttention);
        feed->put_table("feed:sys.cpu.util", {{"seq", "BIGINT"}, {"t_ns", "BIGINT"}, {"value", "FLOAT"}},
                        json{{1, 100, 0.5}, {2, 200, "nan"}, {3, 300, 0.75}});
        ok(*s, "spec.new", {{"id", "live"}});
        ok(*s, "source.add", {{"id", "attn"}, {"catalog", "live:attn.weights"}});
        ok(*s, "source.add", {{"id", "entropy"}, {"from", "attn"},
                              {"transform", {{{"op", "reduce"}, {"axis", -1}, {"fn", "entropy"}},
                                             {{"op", "reduce"}, {"axis", -1}, {"fn", "mean"}}}}});
        ok(*s, "source.add", {{"id", "cpu"}, {"catalog", "feed:sys.cpu.util"}});
        ok(*s, "panel.create", {{"id", "heat"}, {"primitive", "heatmap"}, {"bind", {{"values", "entropy"}}}});
        ok(*s, "panel.create", {{"id", "line"}, {"primitive", "series"},
                                {"bind", {{"x", "cpu.t_ns"}, {"y", {"cpu.value"}}}}});
    }
};

}  // namespace

TEST_CASE("served sources answer resolve.unavailable until their kind has a data source") {
    Fixture f("served-unregistered");
    CHECK(f.s->resolve("heat").error.code == "resolve.unavailable");
    CHECK(f.s->resolve("line").error.code == "resolve.unavailable");
    CHECK(f.s->resolve("heat").generation == 0);
    // The registry takes only the three served kinds.
    CHECK_FALSE(f.s->data_sources().set(CatalogKind::table, f.live));
    CHECK_FALSE(f.s->data_sources().set(CatalogKind::tensor, f.live));
    CHECK(f.s->data_sources().set(CatalogKind::live, f.live));
    CHECK(f.s->data_sources().get(CatalogKind::live) == f.live);
    CHECK(f.s->data_sources().get(CatalogKind::feed) == nullptr);
    CHECK(f.s->resolve("heat").error.ok());
    CHECK(f.s->resolve("line").error.code == "resolve.unavailable");
    CHECK(f.s->data_sources().set(CatalogKind::live, nullptr));
    CHECK(f.s->resolve("heat").error.code == "resolve.unavailable");
}

TEST_CASE("a live tensor takes the spec's transforms: entropy over the last axis, then the mean over tokens") {
    Fixture f("served-entropy");
    f.s->data_sources().set(CatalogKind::live, f.live);
    const PanelData heat = f.s->resolve("heat");
    REQUIRE_MESSAGE(heat.error.ok(), heat.error.message);
    REQUIRE(heat.slots.size() == 1);
    CHECK(heat.slots[0].kind == SlotData::Kind::tensor);
    CHECK(heat.slots[0].tensor.shape() == std::vector<int64_t>{2, 2});
    CHECK(heat.slots[0].tensor.dtype() == DType::f32);
    const std::vector<double> want = expected_entropy(kAttention);
    const std::vector<double> got = values_of(heat.slots[0].tensor);
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < want.size(); ++i) CHECK(got[i] == doctest::Approx(want[i]).epsilon(1e-6));
    CHECK(heat.slots[0].generation == 1);
    // The request carried the session's bounds.
    CHECK(f.live->last_request.max_bytes == ResolveLimits{}.max_tensor_bytes);
}

TEST_CASE("the panel's generation is predicted by panel_generation and changes with each new snapshot") {
    Fixture f("served-generation");
    f.s->data_sources().set(CatalogKind::live, f.live);
    f.s->data_sources().set(CatalogKind::feed, f.feed);
    const spec::Spec spec = f.s->open_spec()->spec;
    DataSources& sources = f.s->data_sources();

    const PanelData first = f.s->resolve("heat");
    REQUIRE(first.error.ok());
    CHECK(first.generation != 0);
    CHECK(panel_generation(spec, "heat", sources) == first.generation);
    // Resolving again with nothing new reads the generation only and reuses
    // the transformed tensor.
    const int reads = f.live->reads;
    const auto before = sources.cache_stats();
    const PanelData again = f.s->resolve("heat");
    CHECK(again.generation == first.generation);
    CHECK(f.live->reads == reads);
    CHECK(sources.cache_stats().reused == before.reused + 1);
    CHECK(sources.cache_stats().computed == before.computed);
    CHECK(values_of(again.slots[0].tensor) == values_of(first.slots[0].tensor));

    // A new snapshot changes the probe before anything is read.
    std::vector<float> next = kAttention;
    next[0] = 9;
    f.live->put_tensor("live:attn.weights", {2, 2, 2, 3}, next);
    CHECK(panel_generation(spec, "heat", sources) != first.generation);
    const PanelData third = f.s->resolve("heat");
    REQUIRE(third.error.ok());
    CHECK(third.generation == panel_generation(spec, "heat", sources));
    CHECK(f.live->reads == reads + 1);
    CHECK(sources.cache_stats().computed == before.computed + 1);
    CHECK(values_of(third.slots[0].tensor)[0] == doctest::Approx(expected_entropy(next)[0]).epsilon(1e-6));

    // An unknown panel has generation 0.
    CHECK(panel_generation(spec, "nothing", sources) == 0);
    // The table panel follows its feed's generation.
    const PanelData line = f.s->resolve("line");
    REQUIRE(line.error.ok());
    CHECK(line.generation == panel_generation(spec, "line", sources));
    f.feed->put_table("feed:sys.cpu.util", {{"seq", "BIGINT"}, {"t_ns", "BIGINT"}, {"value", "FLOAT"}},
                      json{{4, 400, 1.0}});
    CHECK(panel_generation(spec, "line", sources) != line.generation);
}

TEST_CASE("a served table gives columns like a catalog table") {
    Fixture f("served-table");
    f.s->data_sources().set(CatalogKind::feed, f.feed);
    const PanelData line = f.s->resolve("line");
    REQUIRE_MESSAGE(line.error.ok(), line.error.message);
    REQUIRE(line.slots.size() == 2);
    CHECK(line.slots[0].kind == SlotData::Kind::column);
    CHECK(line.slots[0].columns[0].type == "BIGINT");
    CHECK(line.slots[0].values == json{100, 200, 300});
    CHECK(line.slots[1].values == json{0.5, "nan", 0.75});
    CHECK(line.slots[1].generation == 1);
    CHECK(f.feed->last_request.max_rows == ResolveLimits{}.max_rows);
    // A whole-source binding gives every column.
    ok(*f.s, "panel.create", {{"id", "whole"}, {"primitive", "table"}, {"bind", {{"rows", "cpu"}}}});
    const PanelData whole = f.s->resolve("whole");
    REQUIRE(whole.error.ok());
    CHECK(whole.slots[0].columns.size() == 3);
    CHECK(whole.slots[0].values.size() == 3);
}

TEST_CASE("a served source that went away reports it, and the probe follows") {
    Fixture f("served-gone");
    f.s->data_sources().set(CatalogKind::live, f.live);
    const spec::Spec spec = f.s->open_spec()->spec;
    const PanelData before = f.s->resolve("heat");
    REQUIRE(before.error.ok());
    f.live->erase("live:attn.weights");
    const PanelData gone = f.s->resolve("heat");
    CHECK(gone.error.code == "resolve.unavailable");
    CHECK(gone.error.message.find("went away") != std::string::npos);
    CHECK(gone.error.where == "attn");
    CHECK(gone.generation == panel_generation(spec, "heat", f.s->data_sources()));
    CHECK(gone.generation != before.generation);
    // It comes back with a new snapshot.
    f.live->put_tensor("live:attn.weights", {2, 2, 2, 3}, kAttention);
    CHECK(f.s->resolve("heat").error.ok());
}

TEST_CASE("a served tensor past the session's limit is refused, and so is a tensor source that gives a table") {
    SessionOptions options;
    options.registry.check_results = true;
    options.resolve.max_tensor_bytes = 64;
    auto opened = Session::open(fresh_dir("served-limit") + "/workspace", options);
    REQUIRE(opened.error.ok());
    Session& s = *opened.session;
    auto live = std::make_shared<FakeSource>();
    live->put_tensor("live:big", {2, 2, 2, 3}, kAttention);  // 96 bytes
    live->put_table("live:odd", {{"a", "BIGINT"}}, json{{1}});
    s.data_sources().set(CatalogKind::live, live);
    ok(s, "spec.new", {{"id", "lim"}});
    ok(s, "source.add", {{"id", "big"}, {"catalog", "live:big"}});
    ok(s, "source.add", {{"id", "odd"}, {"catalog", "live:odd"}});
    ok(s, "panel.create", {{"id", "a"}, {"primitive", "heatmap"}, {"bind", {{"values", "big"}}}});
    ok(s, "panel.create", {{"id", "b"}, {"primitive", "heatmap"}, {"bind", {{"values", "odd"}}}});
    CHECK(s.resolve("a").error.code == "resolve.too_large");
    CHECK(live->last_request.max_bytes == 64);
    CHECK(s.resolve("b").error.code == "resolve.not_a_tensor");
}

TEST_CASE("the cache of transformed tensors keeps to its bounds and forgets on a new registration") {
    DataSources::Options options;
    options.max_cache_entries = 2;
    options.max_cache_bytes = 1000;
    DataSources sources(options);
    TensorResult small = TensorData::from_bytes(DType::u8, {10}, std::vector<uint8_t>(10, 1));
    TensorResult large = TensorData::from_bytes(DType::u8, {2000}, std::vector<uint8_t>(2000, 1));
    REQUIRE(small.error.ok());
    REQUIRE(large.error.ok());
    TensorData out;
    sources.store_cached("a", 1, small.tensor);
    sources.store_cached("b", 1, small.tensor);
    CHECK(sources.find_cached("a", 1, &out));  // a is now the most recently used
    CHECK_FALSE(sources.find_cached("a", 2, &out));
    sources.store_cached("c", 1, small.tensor);  // evicts b
    CHECK_FALSE(sources.find_cached("b", 1, &out));
    CHECK(sources.find_cached("c", 1, &out));
    CHECK(sources.cache_stats().entries == 2);
    sources.store_cached("d", 1, large.tensor);  // too large to keep
    CHECK_FALSE(sources.find_cached("d", 1, &out));
    CHECK(sources.cache_stats().bytes == 20);
    sources.set(CatalogKind::metric, std::make_shared<FakeSource>());
    CHECK(sources.cache_stats().entries == 0);
    CHECK(sources.cache_stats().bytes == 0);
    CHECK(sources.generation("metric:1/loss") == 0);
    CHECK(sources.generation("not an id") == 0);
}
