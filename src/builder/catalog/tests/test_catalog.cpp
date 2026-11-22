// Tests of Catalog and InMemoryCatalogProvider: merging, duplicate ids,
// failing and faulty providers, sampling limits and the revision.
#include <caliper_builder/catalog/catalog.h>
#include <caliper_builder/catalog/memory_provider.h>

#include <doctest/doctest.h>

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using caliper_builder::Catalog;
using caliper_builder::CatalogColumn;
using caliper_builder::CatalogDtype;
using caliper_builder::CatalogEntry;
using caliper_builder::CatalogFilter;
using caliper_builder::CatalogKind;
using caliper_builder::CatalogSampleRequest;
using caliper_builder::InMemoryCatalogProvider;
using nlohmann::json;
namespace cb = caliper_builder;

namespace {

CatalogEntry table_entry(const std::string& name, std::size_t columns = 2) {
    CatalogEntry e;
    e.kind = CatalogKind::table;
    e.name = name;
    e.id = "table:" + name;
    e.title = name;
    e.origin.database = "workspace";
    e.origin.table = name;
    std::vector<CatalogColumn> cols;
    for (std::size_t i = 0; i < columns; ++i) cols.push_back({"c" + std::to_string(i), "VARCHAR"});
    e.columns = cols;
    return e;
}

CatalogEntry tensor_entry(const std::string& name, std::vector<std::int64_t> shape) {
    CatalogEntry e;
    e.kind = CatalogKind::tensor;
    e.name = name;
    e.id = "tensor:" + name;
    e.title = name;
    e.origin.loader = "weights";
    e.dtype = CatalogDtype::f32;
    e.shape = std::move(shape);
    return e;
}

std::vector<std::uint8_t> floats(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i);
    std::vector<std::uint8_t> out(n * sizeof(float));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

// A provider that throws from every call.
struct ThrowingProvider : cb::CatalogProvider {
    cb::CatalogListResult list(const CatalogFilter&) override { throw std::runtime_error("list exploded"); }
    cb::CatalogDescribeResult describe(const std::string&) override { throw 42; }
    cb::CatalogSampleResult sample(const std::string&, const CatalogSampleRequest&) override {
        throw std::runtime_error("sample exploded");
    }
    std::uint64_t revision() const override { throw std::runtime_error("revision exploded"); }
};

// A provider that breaks the contract: faulty entries and oversized samples.
struct FaultyProvider : cb::CatalogProvider {
    cb::CatalogListResult list(const CatalogFilter&) override {
        cb::CatalogListResult r;
        CatalogEntry bad = table_entry("bad");
        bad.columns.reset();  // a table without columns
        r.entries.push_back(bad);
        CatalogEntry fine = table_entry("fine");
        r.entries.push_back(fine);
        CatalogEntry mangled = table_entry("x");
        mangled.id = std::string("table:\xFF");
        r.entries.push_back(mangled);
        return r;
    }
    cb::CatalogDescribeResult describe(const std::string& id) override {
        cb::CatalogDescribeResult r;
        if (id == "tensor:big") r.entry = tensor_entry("big", {100, 100});
        if (id == "table:other") r.entry = table_entry("not-other");
        return r;
    }
    cb::CatalogSampleResult sample(const std::string& id, const CatalogSampleRequest&) override {
        cb::CatalogSampleResult r;
        r.sample.id = id;
        r.sample.kind = CatalogKind::tensor;
        r.sample.dtype = CatalogDtype::f32;
        r.sample.shape = std::vector<std::int64_t>{100, 100};
        r.sample.stats = cb::CatalogTensorStats{};
        cb::CatalogSlice slice;
        slice.start = {0, 0};
        slice.shape = {100, 100};
        slice.values = json::array();
        for (int i = 0; i < 10000; ++i) slice.values.push_back(0);
        r.sample.slice = slice;
        return r;
    }
    std::uint64_t revision() const override { return 0; }
};

}  // namespace

TEST_CASE("providers merge into one sorted list, and the filter applies to all") {
    auto a = std::make_shared<InMemoryCatalogProvider>();
    auto b = std::make_shared<InMemoryCatalogProvider>();
    REQUIRE(a->put(table_entry("zeta")).ok());
    REQUIRE(a->put(table_entry("alpha")).ok());
    REQUIRE(b->put(tensor_entry("attn", {2, 2})).ok());
    REQUIRE(b->put(table_entry("beta")).ok());
    Catalog catalog;
    REQUIRE(catalog.add_provider("a", a).ok());
    REQUIRE(catalog.add_provider("b", b).ok());

    auto all = catalog.list({});
    REQUIRE(all.error.ok());
    REQUIRE(all.entries.size() == 4);
    CHECK(all.entries[0].id == "table:alpha");
    CHECK(all.entries[1].id == "table:beta");
    CHECK(all.entries[2].id == "table:zeta");
    CHECK(all.entries[3].id == "tensor:attn");
    CHECK_FALSE(all.truncated);
    CHECK(all.warnings.empty());

    CatalogFilter tensors;
    tensors.kind = CatalogKind::tensor;
    CHECK(catalog.list(tensors).entries.size() == 1);

    CatalogFilter prefix;
    prefix.prefix = "be";
    auto some = catalog.list(prefix);
    REQUIRE(some.entries.size() == 1);
    CHECK(some.entries[0].name == "beta");

    CatalogFilter two;
    two.limit = 2;
    auto cut = catalog.list(two);
    CHECK(cut.entries.size() == 2);
    CHECK(cut.truncated);
    CHECK(cut.entries[1].id == "table:beta");
}

TEST_CASE("a duplicate id goes to the higher priority, then to the provider added first") {
    auto low = std::make_shared<InMemoryCatalogProvider>();
    auto high = std::make_shared<InMemoryCatalogProvider>();
    auto later = std::make_shared<InMemoryCatalogProvider>();
    CatalogEntry from_low = table_entry("runs");
    from_low.title = "from low";
    CatalogEntry from_high = table_entry("runs");
    from_high.title = "from high";
    CatalogEntry from_later = table_entry("runs");
    from_later.title = "from later";
    REQUIRE(low->put(from_low).ok());
    REQUIRE(high->put(from_high).ok());
    REQUIRE(later->put(from_later).ok());

    Catalog catalog;
    REQUIRE(catalog.add_provider("low", low, 0).ok());
    REQUIRE(catalog.add_provider("high", high, 10).ok());
    REQUIRE(catalog.add_provider("later", later, 0).ok());
    CHECK(catalog.provider_names() == std::vector<std::string>{"high", "low", "later"});

    const auto merged = catalog.list({});
    REQUIRE(merged.entries.size() == 1);
    CHECK(merged.entries[0].title == "from high");
    REQUIRE(merged.duplicates.size() == 1);
    CHECK(merged.duplicates[0].id == "table:runs");
    CHECK(merged.duplicates[0].kept == "high");
    CHECK(merged.duplicates[0].hidden == std::vector<std::string>{"low", "later"});
    REQUIRE(merged.warnings.size() == 1);
    CHECK(merged.warnings[0] == "table:runs is also offered by low and later; the entry of high is shown.");

    // describe and sample resolve to the same provider as list.
    const auto d = catalog.describe("table:runs");
    REQUIRE(d.entry);
    CHECK(d.entry->title == "from high");

    REQUIRE(catalog.remove_provider("high"));
    CHECK(catalog.describe("table:runs").entry->title == "from low");
}

TEST_CASE("a failing or throwing provider adds a warning and the others still answer") {
    auto good = std::make_shared<InMemoryCatalogProvider>();
    REQUIRE(good->put(table_entry("ok")).ok());
    auto failing = std::make_shared<InMemoryCatalogProvider>();
    failing->fail_with(cb::Error("duckdb.closed", "The database is closed."));
    Catalog catalog;
    REQUIRE(catalog.add_provider("good", good).ok());
    REQUIRE(catalog.add_provider("failing", failing).ok());
    REQUIRE(catalog.add_provider("throwing", std::make_shared<ThrowingProvider>()).ok());

    const auto merged = catalog.list({});
    REQUIRE(merged.entries.size() == 1);
    REQUIRE(merged.warnings.size() == 2);
    CHECK(merged.warnings[0] == "The provider failing failed: The database is closed.");
    CHECK(merged.warnings[1] == "The provider throwing failed: list exploded");

    CHECK(catalog.describe("table:ok").entry);
    // An id no provider has, while one failed, may still exist.
    const auto missing = catalog.describe("table:nowhere");
    CHECK(missing.error.code == "catalog.provider_failed");
    CHECK(catalog.sample("table:nowhere", {}).error.code == "catalog.provider_failed");
    CHECK(catalog.revision() > 0);  // a throwing revision() is skipped
}

TEST_CASE("faulty entries from a provider are dropped with a warning") {
    Catalog catalog;
    REQUIRE(catalog.add_provider("faulty", std::make_shared<FaultyProvider>()).ok());
    const auto merged = catalog.list({});
    REQUIRE(merged.entries.size() == 1);
    CHECK(merged.entries[0].id == "table:fine");
    REQUIRE(merged.warnings.size() == 2);
    CHECK(merged.warnings[0].find("table:bad") != std::string::npos);
    CHECK(merged.warnings[0].find("(columns)") != std::string::npos);
    CHECK(merged.warnings[1].find("an entry with a malformed id") != std::string::npos);
    // The warnings are valid JSON text.
    CHECK_NOTHROW(json(merged.warnings).dump());

    // A described entry of another id, or a slice past 4096 values, is refused.
    CHECK(catalog.describe("table:other").error.code == "catalog.provider_failed");
    const auto big = catalog.sample("tensor:big", {});
    CHECK(big.error.code == "catalog.provider_failed");
}

TEST_CASE("describe and sample check the id first") {
    Catalog catalog;
    REQUIRE(catalog.add_provider("memory", std::make_shared<InMemoryCatalogProvider>()).ok());
    CHECK(catalog.describe("nonsense").error.code == "catalog.invalid_id");
    CHECK(catalog.describe("table:missing").error.code == "catalog.not_found");
    CHECK(catalog.sample("image:x", {}).error.code == "catalog.invalid_id");
    CHECK(catalog.sample("table:missing", {}).error.code == "catalog.not_found");
}

TEST_CASE("a table sample gives its first rows, cuts long cells and makes every cell a scalar") {
    auto memory = std::make_shared<InMemoryCatalogProvider>();
    json rows = json::array();
    for (int i = 0; i < 30; ++i) rows.push_back(json::array({std::to_string(i), i}));
    rows[0][0] = std::string(2000, 'x');
    rows[1][1] = json::object({{"nested", true}});
    rows[2][1] = std::numeric_limits<double>::quiet_NaN();
    CatalogEntry e = table_entry("events");
    (*e.columns)[1].type = "BIGINT";
    e.rows = 30;
    REQUIRE(memory->put_table(e, rows).ok());
    Catalog catalog;
    REQUIRE(catalog.add_provider("memory", memory).ok());

    CatalogSampleRequest request;
    request.limit = 5;
    const auto r = catalog.sample("table:events", request);
    REQUIRE(r.error.ok());
    const json j = cb::catalog_sample_to_json(r.sample);
    CHECK(j["rows"].size() == 5);
    CHECK(j["truncated"] == true);
    CHECK(j["total_rows"] == 30);
    CHECK(j["cut_cells"] == 1);
    CHECK(j["rows"][0][0].get<std::string>().size() == 1024);
    CHECK(j["rows"][1][1] == "{\"nested\":true}");
    CHECK(j["rows"][2][1] == "nan");
    CHECK(j["columns"].size() == 2);
    CHECK(j["dtype"].is_null());
    CHECK(j["stats"].is_null());
    CHECK(j["slice"].is_null());
    CHECK(j.size() == 11);

    CatalogSampleRequest bad;
    bad.limit = 0;
    CHECK(catalog.sample("table:events", bad).error.code == "catalog.bad_request");
    bad.limit = 1001;
    CHECK(catalog.sample("table:events", bad).error.code == "catalog.bad_request");
    CatalogSampleRequest sliced;
    sliced.start = {0};
    CHECK(catalog.sample("table:events", sliced).error.code == "catalog.bad_request");
}

TEST_CASE("a feed sample gives its newest rows, oldest first") {
    auto memory = std::make_shared<InMemoryCatalogProvider>();
    CatalogEntry e;
    e.kind = CatalogKind::feed;
    e.name = "sys.cpu.util";
    e.id = "feed:sys.cpu.util";
    e.title = "CPU Utilization";
    e.columns = std::vector<CatalogColumn>{{"seq", "BIGINT"}, {"t_ns", "BIGINT"}, {"value", "FLOAT"}};
    json rows = json::array();
    for (int i = 1; i <= 10; ++i) rows.push_back(json::array({i, i * 100, 0.5 * i}));
    REQUIRE(memory->put_table(e, rows).ok());
    Catalog catalog;
    REQUIRE(catalog.add_provider("feed", memory).ok());
    CatalogSampleRequest request;
    request.limit = 3;
    const auto r = catalog.sample("feed:sys.cpu.util", request);
    REQUIRE(r.error.ok());
    REQUIRE(r.sample.rows);
    CHECK((*r.sample.rows)[0][0] == 8);
    CHECK((*r.sample.rows)[2][0] == 10);
}

TEST_CASE("a sample stops before 4 MiB of rows") {
    auto memory = std::make_shared<InMemoryCatalogProvider>();
    CatalogEntry e = table_entry("wide", 8);
    json rows = json::array();
    const std::string cell(1024, 'w');
    for (int i = 0; i < 1000; ++i) rows.push_back(json(std::vector<std::string>(8, cell)));
    REQUIRE(memory->put_table(e, rows).ok());
    Catalog catalog;
    REQUIRE(catalog.add_provider("memory", memory).ok());
    CatalogSampleRequest request;
    request.limit = 1000;
    const auto r = catalog.sample("table:wide", request);
    REQUIRE(r.error.ok());
    CHECK(r.sample.truncated);
    CHECK(r.sample.rows->size() < 1000);
    CHECK(r.sample.rows->dump().size() <= cb::CatalogSampleLimits::max_sample_json);
}

TEST_CASE("a tensor sample through the catalog, and a live view with no data") {
    auto memory = std::make_shared<InMemoryCatalogProvider>();
    REQUIRE(memory->put_tensor(tensor_entry("w", {3, 4}), floats(12)).ok());
    CHECK(memory->put_tensor(tensor_entry("short", {3, 4}), floats(11)).code == "catalog.bad_tensor");
    CHECK(memory->put_table(tensor_entry("t", {1}), json::array()).code == "catalog.bad_request");

    CatalogEntry live;
    live.kind = CatalogKind::live;
    live.name = "attn.weights";
    live.id = "live:attn.weights";
    live.title = "attn.weights";
    REQUIRE(memory->put(live).ok());
    Catalog catalog;
    REQUIRE(catalog.add_provider("memory", memory).ok());

    CatalogSampleRequest request;
    request.start = {1, 1};
    request.count = {2, 2};
    const auto r = catalog.sample("tensor:w", request);
    REQUIRE(r.error.ok());
    const json j = cb::catalog_sample_to_json(r.sample);
    CHECK(j["dtype"] == "f32");
    CHECK(j["shape"] == json::array({3, 4}));
    CHECK(j["stats"]["count"] == 12);
    CHECK(j["stats"]["exact"] == true);
    CHECK(j["stats"]["max"] == 11.0);
    CHECK(j["slice"]["values"] == json::array({5.0, 6.0, 9.0, 10.0}));
    CHECK(j["columns"].is_null());
    CHECK(j["rows"].is_null());

    request.count = {3, 2};
    CHECK(catalog.sample("tensor:w", request).error.code == "catalog.bad_request");
    CHECK(catalog.sample("live:attn.weights", {}).error.code == "catalog.no_data");
}

TEST_CASE("the revision grows on every schema change and never drops") {
    auto memory = std::make_shared<InMemoryCatalogProvider>();
    Catalog catalog;
    const auto r0 = catalog.revision();
    REQUIRE(catalog.add_provider("memory", memory).ok());
    const auto r1 = catalog.revision();
    CHECK(r1 > r0);
    REQUIRE(memory->put(table_entry("t")).ok());
    const auto r2 = catalog.revision();
    CHECK(r2 > r1);
    CatalogEntry same = table_entry("t");
    same.rows = 99;  // data only
    REQUIRE(memory->put(same).ok());
    CHECK(catalog.revision() == r2);
    REQUIRE(memory->put(table_entry("t", 3)).ok());  // one more column
    const auto r3 = catalog.revision();
    CHECK(r3 > r2);
    REQUIRE(memory->remove("table:t"));
    const auto r4 = catalog.revision();
    CHECK(r4 > r3);
    REQUIRE(catalog.remove_provider("memory"));
    CHECK(catalog.revision() > r4);
}

TEST_CASE("providers are checked when they are added") {
    Catalog catalog;
    CHECK(catalog.add_provider("x", nullptr).code == "catalog.bad_provider");
    CHECK(catalog.add_provider("Bad", std::make_shared<InMemoryCatalogProvider>()).code == "catalog.bad_provider");
    CHECK(catalog.add_provider("", std::make_shared<InMemoryCatalogProvider>()).code == "catalog.bad_provider");
    REQUIRE(catalog.add_provider("p0", std::make_shared<InMemoryCatalogProvider>()).ok());
    CHECK(catalog.add_provider("p0", std::make_shared<InMemoryCatalogProvider>()).code == "catalog.bad_provider");
    for (int i = 1; i < 32; ++i)
        REQUIRE(catalog.add_provider("p" + std::to_string(i), std::make_shared<InMemoryCatalogProvider>()).ok());
    CHECK(catalog.add_provider("p32", std::make_shared<InMemoryCatalogProvider>()).code == "catalog.bad_provider");
    CHECK_FALSE(catalog.remove_provider("nobody"));

    CatalogFilter filter;
    filter.limit = 0;
    CHECK(catalog.list(filter).error.code == "catalog.bad_request");
    filter.limit = 1001;
    CHECK(catalog.list(filter).error.code == "catalog.bad_request");
    filter.limit = 10;
    filter.prefix = std::string(300, 'p');
    CHECK(catalog.list(filter).error.code == "catalog.bad_request");
}
