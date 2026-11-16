// Tests of CatalogEntry: its strict JSON form, the id grammar and every rule
// of docs/plans/catalog.md, with malformed and hostile entries.
#include <caliper_builder/catalog/catalog_entry.h>

#include <doctest/doctest.h>

#include <string>

using caliper_builder::CatalogColumn;
using caliper_builder::CatalogDtype;
using caliper_builder::CatalogEntry;
using caliper_builder::CatalogKind;
using nlohmann::json;
namespace cb = caliper_builder;

namespace {

// The example entry of the contracts.
json tensor_json() {
    return json::parse(R"({
      "id": "tensor:attn_weights", "kind": "tensor", "name": "attn_weights", "title": "attn_weights",
      "origin": {"loader": "run42-weights", "file": "model.safetensors"},
      "columns": null, "rows": null, "dtype": "f32", "shape": [12, 12, 64, 64], "rate_hz": null,
      "bytes": 28311552, "wire": true, "warnings": [], "updated_at": "2026-11-15T12:00:00Z"})");
}

json table_json() {
    return json::parse(R"({
      "id": "table:run42_metrics", "kind": "table", "name": "run42_metrics", "title": "Run 42 metrics",
      "origin": {"loader": "run42-metrics", "file": "data/run42/*.parquet", "output": "run42_metrics"},
      "columns": [{"name": "step", "type": "BIGINT"}, {"name": "loss", "type": "DOUBLE"}],
      "rows": 1200, "dtype": null, "shape": null, "rate_hz": null, "bytes": null, "wire": true,
      "warnings": ["loaded weights-only from a pickle file"], "updated_at": "2026-11-15T12:00:00.250Z"})");
}

// Parses `j` and returns the error's `where`, or "ok".
std::string fault_of(const json& j) {
    const auto parsed = cb::catalog_entry_from_json(j);
    if (!parsed.error) return "ok";
    CHECK(parsed.error.code == "catalog.invalid_entry");
    CHECK(!parsed.error.message.empty());
    return parsed.error.where;
}

CatalogEntry live_entry() {
    CatalogEntry e;
    e.kind = CatalogKind::live;
    e.name = "attn.weights";
    e.id = cb::make_catalog_id(e.kind, e.name);
    e.title = e.name;
    e.origin.session = "train";
    e.origin.view = "attn.weights";
    e.origin.view_type = "heatmap";
    e.wire = true;
    return e;
}

}  // namespace

TEST_CASE("the example entry of the contracts parses and writes back unchanged") {
    const json j = tensor_json();
    const auto parsed = cb::catalog_entry_from_json(j);
    REQUIRE(parsed.error.ok());
    const CatalogEntry& e = parsed.entry;
    CHECK(e.kind == CatalogKind::tensor);
    CHECK(e.name == "attn_weights");
    CHECK(e.dtype == CatalogDtype::f32);
    REQUIRE(e.shape);
    CHECK(*e.shape == std::vector<std::int64_t>{12, 12, 64, 64});
    CHECK(e.origin.loader == std::string("run42-weights"));
    CHECK(!e.origin.output);
    CHECK(e.bytes == std::uint64_t(28311552));
    CHECK(cb::catalog_entry_to_json(e) == j);
}

TEST_CASE("a table entry round-trips with its columns, rows and warnings") {
    const json j = table_json();
    const auto parsed = cb::catalog_entry_from_json(j);
    REQUIRE(parsed.error.ok());
    REQUIRE(parsed.entry.columns);
    CHECK(parsed.entry.columns->size() == 2);
    CHECK((*parsed.entry.columns)[1].type == "DOUBLE");
    CHECK(parsed.entry.rows == std::uint64_t(1200));
    CHECK(cb::catalog_entry_to_json(parsed.entry) == j);
    // The JSON form always holds all fourteen keys.
    CHECK(cb::catalog_entry_to_json(parsed.entry).size() == 14);
}

TEST_CASE("each kind follows its own rules") {
    CatalogEntry live = live_entry();
    CHECK(cb::validate_catalog_entry(live).ok());  // before its first snapshot
    live.dtype = CatalogDtype::f32;
    CHECK(cb::validate_catalog_entry(live).where == "shape");
    live.shape = std::vector<std::int64_t>{4, 8};
    live.rate_hz = 60.0;
    CHECK(cb::validate_catalog_entry(live).ok());

    CatalogEntry feed;
    feed.kind = CatalogKind::feed;
    feed.name = "sys.cpu.util";
    feed.id = "feed:sys.cpu.util";
    feed.title = "CPU Utilization";
    feed.origin.service = "feed";
    feed.origin.channel = "sys.cpu.util";
    feed.origin.units = "%";
    feed.columns = std::vector<CatalogColumn>{{"seq", "BIGINT"}, {"t_ns", "BIGINT"}, {"value", "FLOAT"}};
    feed.rate_hz = 10.0;
    CHECK(cb::validate_catalog_entry(feed).ok());

    CatalogEntry metric;
    metric.kind = CatalogKind::metric;
    metric.name = "7/train/loss";
    metric.id = "metric:7/train/loss";
    metric.title = "run42 train/loss";
    metric.origin.service = "metrics";
    metric.origin.run = 7;
    metric.origin.tag = "train/loss";
    metric.columns = std::vector<CatalogColumn>{{"step", "BIGINT"}, {"value", "DOUBLE"}};
    CHECK(cb::validate_catalog_entry(metric).ok());
    metric.rate_hz = 1.0;
    CHECK(cb::validate_catalog_entry(metric).where == "rate_hz");

    // A wav tensor carries its sampling rate.
    auto wav = cb::catalog_entry_from_json(tensor_json()).entry;
    wav.rate_hz = 48000.0;
    CHECK(cb::validate_catalog_entry(wav).ok());
    const auto back = cb::catalog_entry_from_json(cb::catalog_entry_to_json(wav));
    REQUIRE(back.error.ok());
    CHECK(back.entry.rate_hz == 48000.0);
}

TEST_CASE("a malformed entry names the key at fault") {
    CHECK(fault_of(json::array()) == "");
    CHECK(fault_of(json("tensor:x")) == "");

    json j = tensor_json();
    j.erase("updated_at");
    CHECK(fault_of(j) == "updated_at");

    j = tensor_json();
    j["extra"] = 1;
    CHECK(fault_of(j) == "extra");

    j = tensor_json();
    j["id"] = 7;
    CHECK(fault_of(j) == "id");

    j = tensor_json();
    j["id"] = "tensor:other";
    CHECK(fault_of(j) == "id");

    j = tensor_json();
    j["kind"] = "image";
    CHECK(fault_of(j) == "kind");

    j = tensor_json();
    j["name"] = "";
    j["id"] = "tensor:";
    CHECK(fault_of(j) == "name");

    j = tensor_json();
    j["title"] = " leading space";
    CHECK(fault_of(j) == "title");

    j = tensor_json();
    j["title"] = std::string("tab\there");
    CHECK(fault_of(j) == "title");

    j = tensor_json();
    j["origin"]["path"] = "x";
    CHECK(fault_of(j) == "origin.path");

    j = tensor_json();
    j["origin"]["file"] = 3;
    CHECK(fault_of(j) == "origin.file");

    j = tensor_json();
    j["origin"]["run"] = -1;
    CHECK(fault_of(j) == "origin.run");

    j = tensor_json();
    j["origin"] = "loader";
    CHECK(fault_of(j) == "origin");

    j = tensor_json();
    j["columns"] = json::array({{{"name", "a"}, {"type", "INT"}}});
    CHECK(fault_of(j) == "columns");

    j = table_json();
    j["columns"] = nullptr;
    CHECK(fault_of(j) == "columns");

    j = table_json();
    j["columns"][0]["nullable"] = true;
    CHECK(fault_of(j) == "columns[0].nullable");

    j = table_json();
    j["columns"][1]["type"] = "";
    CHECK(fault_of(j) == "columns[1].type");

    j = table_json();
    j["shape"] = json::array({3});
    CHECK(fault_of(j) == "shape");

    j = table_json();
    j["rate_hz"] = 2.0;
    CHECK(fault_of(j) == "rate_hz");

    j = tensor_json();
    j["dtype"] = "f8";
    CHECK(fault_of(j) == "dtype");

    j = tensor_json();
    j["dtype"] = nullptr;
    CHECK(fault_of(j) == "dtype");

    j = tensor_json();
    j["rows"] = 4;
    CHECK(fault_of(j) == "rows");

    j = tensor_json();
    j["rate_hz"] = 0;
    CHECK(fault_of(j) == "rate_hz");

    j = tensor_json();
    j["rate_hz"] = -5.0;
    CHECK(fault_of(j) == "rate_hz");

    j = tensor_json();
    j["rate_hz"] = 2e9;
    CHECK(fault_of(j) == "rate_hz");

    j = tensor_json();
    j["bytes"] = -1;
    CHECK(fault_of(j) == "bytes");

    j = tensor_json();
    j["bytes"] = 1.5;
    CHECK(fault_of(j) == "bytes");

    j = tensor_json();
    j["wire"] = "yes";
    CHECK(fault_of(j) == "wire");

    j = tensor_json();
    j["warnings"] = json::array({1});
    CHECK(fault_of(j) == "warnings[0]");

    j = tensor_json();
    j["updated_at"] = "2026-11-15 12:00:00";
    CHECK(fault_of(j) == "updated_at");

    j = tensor_json();
    j["updated_at"] = "2026-13-15T12:00:00Z";
    CHECK(fault_of(j) == "updated_at");
}

TEST_CASE("entries at and beyond their limits") {
    json j = tensor_json();
    j["shape"] = json::array({1, 1, 1, 1, 1, 1, 1, 1});
    j["bytes"] = 4;
    CHECK(fault_of(j) == "ok");
    j["shape"] = json::array({1, 1, 1, 1, 1, 1, 1, 1, 1});
    CHECK(fault_of(j) == "shape");

    j = tensor_json();
    j["shape"] = json::array({-1, 4});
    CHECK(fault_of(j) == "shape[0]");

    j = tensor_json();  // 2^31 * 2^31 * 4 passes 2^62
    j["shape"] = json::array({2147483648LL, 2147483648LL, 4});
    CHECK(fault_of(j) == "shape");

    j = tensor_json();  // a zero axis makes any product fine
    j["shape"] = json::array({0, 4611686018427387904LL});
    CHECK(fault_of(j) == "ok");

    j = tensor_json();  // 2^64 does not fit a row count
    j["bytes"] = json::parse("18446744073709551616");
    CHECK(fault_of(j) == "bytes");

    j = table_json();
    j["columns"] = json::array();
    for (int i = 0; i < 4097; ++i) j["columns"].push_back({{"name", "c" + std::to_string(i)}, {"type", "INT"}});
    CHECK(fault_of(j) == "columns");
    j["columns"].erase(j["columns"].size() - 1);
    CHECK(fault_of(j) == "ok");

    j = tensor_json();
    j["warnings"] = json::array();
    for (int i = 0; i < 17; ++i) j["warnings"].push_back("w");
    CHECK(fault_of(j) == "warnings");

    j = tensor_json();
    j["warnings"] = json::array({std::string(1025, 'w')});
    CHECK(fault_of(j) == "warnings[0]");

    j = tensor_json();
    const std::string long_name(257, 'a');
    j["name"] = long_name;
    j["id"] = "tensor:" + long_name;
    CHECK(fault_of(j) == "name");
    j["name"] = long_name.substr(1);
    j["id"] = "tensor:" + long_name.substr(1);
    CHECK(fault_of(j) == "ok");

    j = tensor_json();
    j["origin"]["file"] = std::string(1025, 'f');
    CHECK(fault_of(j) == "origin.file");
}

TEST_CASE("a struct with invalid UTF-8 is refused, so its JSON never throws") {
    CatalogEntry e = live_entry();
    e.title = std::string("bad \xC3\x28 byte");
    CHECK(cb::validate_catalog_entry(e).where == "title");
    e.title = "ok";
    e.origin.view = std::string("\xED\xA0\x80");  // a surrogate
    CHECK(cb::validate_catalog_entry(e).where == "origin.view");
    e.origin.view = std::string("\xC0\xAF");  // an overlong slash
    CHECK(cb::validate_catalog_entry(e).where == "origin.view");
    e.origin.view = "attn \xE2\x80\x94 weights";  // an em dash is fine
    CHECK(cb::validate_catalog_entry(e).ok());
}

TEST_CASE("source ids split at the first colon and check both parts") {
    auto p = cb::parse_catalog_id("live:attn.weights");
    REQUIRE(p.error.ok());
    CHECK(p.kind == CatalogKind::live);
    CHECK(p.name == "attn.weights");

    p = cb::parse_catalog_id("metric:7/train/loss");
    REQUIRE(p.error.ok());
    CHECK(p.name == "7/train/loss");

    p = cb::parse_catalog_id("table:a:b");
    REQUIRE(p.error.ok());
    CHECK(p.name == "a:b");

    for (const char* bad : {"", "tensor", ":x", "image:x", "Table:x", "tensor:", "tensor: x", "tensor:x ",
                            "tensor:a\nb"}) {
        const auto q = cb::parse_catalog_id(bad);
        CHECK(q.error.code == "catalog.invalid_id");
    }
    CHECK(cb::parse_catalog_id("tensor:" + std::string(256, 'n')).error.ok());
    CHECK(cb::parse_catalog_id("tensor:" + std::string(257, 'n')).error.code == "catalog.invalid_id");
}

TEST_CASE("timestamps are UTC ISO 8601") {
    CHECK(cb::is_utc_timestamp("2026-11-15T12:00:00Z"));
    CHECK(cb::is_utc_timestamp("2026-11-15T12:00:00.123456789Z"));
    CHECK(cb::is_utc_timestamp("2016-12-31T23:59:60Z"));
    CHECK_FALSE(cb::is_utc_timestamp("2026-11-15T12:00:00"));
    CHECK_FALSE(cb::is_utc_timestamp("2026-11-15T12:00:00+00:00"));
    CHECK_FALSE(cb::is_utc_timestamp("2026-11-15T24:00:00Z"));
    CHECK_FALSE(cb::is_utc_timestamp("2026-11-15T12:00:00.Z"));
    CHECK_FALSE(cb::is_utc_timestamp("2026-11-15T12:00:00.1234567890Z"));
    CHECK_FALSE(cb::is_utc_timestamp("26-11-15T12:00:00Z"));
}

TEST_CASE("kinds and dtypes have their names and sizes") {
    CHECK(std::string(cb::catalog_kind_name(CatalogKind::metric)) == "metric");
    CHECK(cb::catalog_kind_from_name("feed") == CatalogKind::feed);
    CHECK(!cb::catalog_kind_from_name("FEED"));
    CHECK(cb::catalog_kind_is_tabular(CatalogKind::feed));
    CHECK_FALSE(cb::catalog_kind_is_tabular(CatalogKind::live));
    CHECK(std::string(cb::catalog_dtype_name(CatalogDtype::boolean)) == "bool");
    CHECK(cb::catalog_dtype_from_name("bool") == CatalogDtype::boolean);
    CHECK(cb::catalog_dtype_size(CatalogDtype::f16) == 2);
    CHECK(cb::catalog_dtype_size(CatalogDtype::u64) == 8);
}
