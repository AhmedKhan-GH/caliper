// DataStore — SQL over the host store, results out as Arrow C streams
// (caliper.data.v1, PLATFORM.md §7.7, §16 contract): a known table queried
// back yields exact rows through the stream; bad SQL fails with a reason,
// never a crash; registered datasets round-trip.
#include <doctest/doctest.h>

#include "data_store.h"

#include <caliper/arrow_c.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using caliper_host::DataStore;

namespace {

// Drain an ArrowArrayStream of one INT32 column + one DOUBLE column into
// vectors. Exercises the full consumer protocol: get_schema, get_next until
// end-of-stream (released empty array), release everything exactly once.
struct TwoCol {
    std::vector<int32_t> a;
    std::vector<double>  b;
    int64_t schema_children = -1;
};

TwoCol drain_i32_f64(ArrowArrayStream* stream) {
    TwoCol out;

    ArrowSchema schema = {};
    REQUIRE(stream->get_schema(stream, &schema) == 0);
    out.schema_children = schema.n_children;
    // Column 0's format: "i" = int32, "l" = int64 (Arrow format strings) —
    // e.g. read_csv_auto infers BIGINT where a literal column is INTEGER.
    bool a_is_i64 = false;
    if (schema.n_children >= 1)
        a_is_i64 = std::string(schema.children[0]->format) == "l";
    if (schema.release) schema.release(&schema);

    for (;;) {
        ArrowArray array = {};
        REQUIRE(stream->get_next(stream, &array) == 0);
        if (!array.release) break;  // end of stream (spec)
        REQUIRE(array.n_children >= 2);
        const ArrowArray* ca = array.children[0];
        const ArrowArray* cb = array.children[1];
        // buffers[0] = validity, buffers[1] = data (primitive layout)
        const double* db = static_cast<const double*>(cb->buffers[1]);
        for (int64_t i = 0; i < array.length; i++) {
            if (a_is_i64)
                out.a.push_back(static_cast<int32_t>(
                    static_cast<const int64_t*>(ca->buffers[1])[ca->offset + i]));
            else
                out.a.push_back(
                    static_cast<const int32_t*>(ca->buffers[1])[ca->offset + i]);
            out.b.push_back(db[cb->offset + i]);
        }
        array.release(&array);
    }
    stream->release(stream);
    CHECK(stream->release == nullptr);  // release must null itself (spec)
    return out;
}

}  // namespace

TEST_CASE("data_store: query round-trips exact rows through Arrow") {
    DataStore store;
    REQUIRE(store.open(":memory:"));

    REQUIRE(store.exec("CREATE TABLE t(a INTEGER, b DOUBLE)"));
    REQUIRE(store.exec("INSERT INTO t VALUES (1, 0.5), (2, 1.5), (3, 2.5)"));

    ArrowArrayStream stream = {};
    REQUIRE(store.query("SELECT a, b FROM t ORDER BY a", &stream));

    TwoCol got = drain_i32_f64(&stream);
    CHECK(got.schema_children == 2);
    REQUIRE(got.a.size() == 3);
    CHECK(got.a == std::vector<int32_t>({1, 2, 3}));
    CHECK(got.b == std::vector<double>({0.5, 1.5, 2.5}));
}

TEST_CASE("data_store: bad SQL fails with a reason, no crash") {
    DataStore store;
    REQUIRE(store.open(":memory:"));

    ArrowArrayStream stream = {};
    CHECK_FALSE(store.query("SELECT FROM garbage syntax here", &stream));
    CHECK_FALSE(store.last_error().empty());
    // The stream was never filled; the consumer must not need to release it.
    CHECK(stream.release == nullptr);
}

TEST_CASE("data_store: register + open a csv dataset round-trips") {
    fs::path csv = fs::temp_directory_path() /
                   ("caliper-data-test-" + std::to_string(::getpid()) + ".csv");
    {
        std::ofstream f(csv);
        f << "a,b\n10,0.25\n20,0.75\n";
    }

    DataStore store;
    REQUIRE(store.open(":memory:"));
    REQUIRE(store.register_dataset("points", csv.string()));

    ArrowArrayStream stream = {};
    REQUIRE(store.open_dataset("points", &stream));
    TwoCol got = drain_i32_f64(&stream);
    REQUIRE(got.a.size() == 2);
    CHECK(got.a == std::vector<int32_t>({10, 20}));
    CHECK(got.b == std::vector<double>({0.25, 0.75}));

    // Unknown dataset: inert failure with a reason.
    ArrowArrayStream s2 = {};
    CHECK_FALSE(store.open_dataset("no-such", &s2));
    CHECK_FALSE(store.last_error().empty());

    std::error_code ec;
    fs::remove(csv, ec);
}

TEST_CASE("data_store: invalid dataset names are rejected, not injected") {
    DataStore store;
    REQUIRE(store.open(":memory:"));
    CHECK_FALSE(store.register_dataset("bad name; DROP TABLE x", "t"));
    CHECK_FALSE(store.last_error().empty());
}

TEST_CASE("data_store: unopened store is inert") {
    DataStore store;
    ArrowArrayStream stream = {};
    CHECK_FALSE(store.query("SELECT 1", &stream));
    CHECK_FALSE(store.register_dataset("x", "y"));
    CHECK_FALSE(store.last_error().empty());
}
